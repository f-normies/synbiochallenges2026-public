"""DAG driver: topo-sort the stage specs, run each in its env, support resume.

Config is a plain dict (Hydra -> OmegaConf.to_container). Edges are derived from
artifact keys: stage B depends on A iff some B.input is an A.output. Keys not
produced by any stage are external roots, resolved from cfg["artifacts"].
"""

import hashlib
import json
import time
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Any

from synbio.utils.logging import get_logger

from . import STAGE_REGISTRY, StageFactory, discover_stages
from .executor import Executor, LocalExecutor
from .manifest import RunManifest, StageNode, env_snapshot, hash_path
from .stage import StageConfig, StageResult, StageSpec

__all__ = ["run_pipeline", "topo_sort", "resolve_inputs"]

logger = get_logger(__name__)


def topo_sort(specs: list[StageSpec], external: set[str]) -> list[str]:
    """Return stage names in dependency order (producers before consumers)."""
    producer: dict[str, str] = {}
    for spec in specs:
        for key in spec.outputs:
            producer[key] = spec.name
    graph: dict[str, set[str]] = {}
    for spec in specs:
        deps = set()
        for key in spec.inputs:
            if key in producer:
                deps.add(producer[key])
            elif key not in external:
                raise ValueError(f"{spec.name}: input '{key}' has no producer and is not external")
        graph[spec.name] = deps
    return list(TopologicalSorter(graph).static_order())


def resolve_inputs(
    spec: StageSpec,
    nodes: dict[str, StageNode],
    artifacts: dict[str, str],
    runs_dir: Path,
    run_id: str,
    order: list[str],
    full_producer: dict[str, str],
) -> dict[str, str]:
    """Map each input key to an absolute path.

    Resolution order per key: external root (cfg artifacts) -> produced by a stage
    that ran this invocation (in-memory node) -> produced by a stage in a prior run
    (read its persisted result.json from the deterministic stage dir).
    """
    in_run_producer: dict[str, str] = {}
    for name, node in nodes.items():
        for key in node.outputs:
            in_run_producer[key] = name
    resolved: dict[str, str] = {}
    for key in spec.inputs:
        if key in artifacts:
            resolved[key] = str(Path(artifacts[key]).resolve())
        elif key in in_run_producer:
            prod = in_run_producer[key]
            stage_dir = _stage_dir(runs_dir, run_id, prod, order)
            resolved[key] = str((stage_dir / nodes[prod].outputs[key]).resolve())
        elif key in full_producer:
            prod = full_producer[key]
            stage_dir = _stage_dir(runs_dir, run_id, prod, order)
            result_path = stage_dir / "result.json"
            if not result_path.exists():
                raise ValueError(
                    f"{spec.name}: input '{key}' is produced by '{prod}', which has not "
                    f"run (no {result_path}); run it first or include it in stages"
                )
            prod_outputs = StageResult.read_json(result_path).outputs
            resolved[key] = str((stage_dir / prod_outputs[key]).resolve())
        else:
            raise ValueError(f"{spec.name}: cannot resolve input '{key}'")
    return resolved


def _safe(name: str) -> str:
    return name.replace(".", "_")


def _stage_dir(runs_dir: Path, run_id: str, name: str, order: list[str]) -> Path:
    idx = order.index(name)
    return runs_dir / run_id / f"{idx:02d}_{_safe(name)}"


def _stage_params(params_all: dict[str, Any], name: str) -> dict[str, Any]:
    """Per-stage params: exact dotted key, else parent group (e.g. score_stability)."""
    if name in params_all:
        return params_all[name]
    return params_all.get(name.split(".")[0], {})


def _config_hash(spec: StageSpec, params: dict[str, Any], inputs: dict[str, str]) -> str:
    payload = {
        "name": spec.name,
        "env": spec.env,
        "inputs": spec.inputs,
        "outputs": spec.outputs,
        "params": params,
        "input_paths": inputs,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def run_pipeline(
    cfg: dict[str, Any],
    executor: Executor | None = None,
    force: bool = False,
) -> RunManifest:
    """Execute the configured stages in dependency order, returning the manifest."""
    discover_stages()
    external = set(cfg["artifacts"].keys())
    selected = cfg["stages"]
    order = topo_sort([StageFactory(n) for n in STAGE_REGISTRY], external)
    run_order = [n for n in order if n in selected]

    runs_dir = Path(cfg["runs_dir"])
    run_id = cfg["run_id"]
    seed = int(cfg.get("seed", 42))
    params_all = cfg.get("params", {})
    if executor is None:
        executor = LocalExecutor(repo_src=str(Path(__file__).resolve().parents[2]))

    manifest = RunManifest(
        run_id=run_id,
        git_sha=cfg.get("git_sha", ""),
        config_hash=hashlib.sha256(
            json.dumps(cfg, sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
        seed=seed,
        env_snapshot=env_snapshot(),
        nodes=[],
    )
    full_producer: dict[str, str] = {}
    for n in STAGE_REGISTRY:
        for key in StageFactory(n).outputs:
            full_producer[key] = n

    nodes: dict[str, StageNode] = {}
    manifest_path = runs_dir / run_id / "manifest.json"

    for name in run_order:
        spec = StageFactory(name)
        stage_dir = _stage_dir(runs_dir, run_id, name, order)
        stage_dir.mkdir(parents=True, exist_ok=True)
        inputs = resolve_inputs(
            spec, nodes, cfg["artifacts"], runs_dir, run_id, order, full_producer
        )
        input_hashes = {k: hash_path(v) for k, v in inputs.items()}
        cfg_hash = _config_hash(spec, _stage_params(params_all, name), inputs)

        node = _maybe_skip(stage_dir, spec, cfg_hash, input_hashes, force)
        if node is not None:
            logger.info("skip %s (up to date)", name)
            nodes[name] = node
            manifest.nodes.append(node)
            manifest.write_json(manifest_path)
            continue

        resolved = StageConfig(
            name=name, stage_dir=str(stage_dir), seed=seed,
            inputs=inputs, params=_stage_params(params_all, name),
        )
        resolved.write_yaml(stage_dir / "resolved.yaml")
        cmd = [
            "python", "-m", f"synbio.stages.{spec.module}",
            "run", "--spec", name, "--cfg", str(stage_dir / "resolved.yaml"),
        ]
        logger.info("run %s (env=%s)", name, spec.env)
        start = time.time()
        exec_result = executor.run(cmd, env=spec.env, log_path=stage_dir / "log.txt")
        duration = time.time() - start
        if exec_result.returncode != 0:
            raise RuntimeError(f"stage {name} failed (rc={exec_result.returncode}); see {stage_dir/'log.txt'}")

        result = StageResult.read_json(stage_dir / "result.json")
        if set(result.outputs) != set(spec.outputs):
            raise ValueError(
                f"stage {name}: outputs {set(result.outputs)} != declared {set(spec.outputs)}"
            )
        output_hashes = {
            k: hash_path(stage_dir / v) for k, v in result.outputs.items()
        }
        (stage_dir / "state.json").write_text(
            json.dumps({"config_hash": cfg_hash, "input_hashes": input_hashes})
        )
        node = StageNode(
            stage=name, env=spec.env, status="completed",
            cmd="synrun " + spec.env + " " + " ".join(cmd),
            seed=seed, duration_s=duration,
            input_hashes=input_hashes, output_hashes=output_hashes,
            outputs=dict(result.outputs),
            n_in=result.metrics.get("n_in"), n_out=result.metrics.get("n_out"),
            decisions=result.decisions,
        )
        nodes[name] = node
        manifest.nodes.append(node)
        manifest.write_json(manifest_path)

    return manifest


def _maybe_skip(
    stage_dir: Path,
    spec: StageSpec,
    cfg_hash: str,
    input_hashes: dict[str, str],
    force: bool,
) -> StageNode | None:
    """Return a reusable StageNode if the stage is up to date (spec §6.3).

    Up to date means: result.json exists & completed, declared outputs match, and
    BOTH the config hash and every input hash match the persisted state.json.
    """
    if force:
        return None
    result_path = stage_dir / "result.json"
    state_path = stage_dir / "state.json"
    if not (result_path.exists() and state_path.exists()):
        return None
    state = json.loads(state_path.read_text())
    if state.get("config_hash") != cfg_hash:
        return None
    if state.get("input_hashes") != input_hashes:
        return None
    result = StageResult.read_json(result_path)
    if result.status != "completed" or set(result.outputs) != set(spec.outputs):
        return None
    if not all((stage_dir / v).exists() for v in result.outputs.values()):
        return None
    output_hashes = {k: hash_path(stage_dir / v) for k, v in result.outputs.items()}
    return StageNode(
        stage=spec.name, env=spec.env, status="completed",
        cmd="(skipped: up to date)", seed=0, duration_s=0.0,
        input_hashes=input_hashes, output_hashes=output_hashes,
        outputs=dict(result.outputs),
        n_in=result.metrics.get("n_in"), n_out=result.metrics.get("n_out"),
        decisions=result.decisions,
    )
