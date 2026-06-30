"""Stage contract: the import-safe types shared by orchestrator and workers.

A stage module defines one or more `StageSpec`s plus heavy `run()` functions that
import torch/esm/etc. lazily *inside* the body. The orchestrator imports only the
specs. The runner hands a worker a `StageConfig` (resolved.yaml); the worker hands
back a `StageResult` (result.json).
"""

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "StageSpec",
    "Decision",
    "StageResult",
    "StageConfig",
    "cli",
]


@dataclass(frozen=True)
class StageSpec:
    """Immutable description of one DAG node (import-safe; no heavy deps)."""

    name: str  # dotted unique id, e.g. "generate.ligandmpnn"
    module: str  # python module under synbio.stages, e.g. "generate"
    env: str  # micromamba env name
    inputs: tuple[str, ...] = ()  # input artifact keys
    outputs: tuple[str, ...] = ()  # output artifact keys


@dataclass(frozen=True)
class Decision:
    """One recorded funnel decision for the manifest / PDF decision tree."""

    name: str
    threshold: str = ""
    kept: int | None = None
    dropped: int | None = None
    note: str = ""


@dataclass
class StageResult:
    """What a worker returns to the runner, serialized as result.json."""

    outputs: dict[str, str]  # artifact key -> path (relative to stage_dir or abs)
    decisions: list[Decision] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "outputs": self.outputs,
            "decisions": [asdict(d) for d in self.decisions],
            "metrics": self.metrics,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StageResult":
        return cls(
            outputs=dict(d["outputs"]),
            decisions=[Decision(**x) for x in d.get("decisions", [])],
            metrics=dict(d.get("metrics", {})),
            status=d.get("status", "completed"),
        )

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def read_json(cls, path: str | Path) -> "StageResult":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class StageConfig:
    """Resolved per-stage config the runner writes and the worker reads."""

    name: str
    stage_dir: str
    seed: int
    inputs: dict[str, str]  # resolved input key -> absolute path
    params: dict[str, Any]  # stage parameters (thresholds, temps, ...)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StageConfig":
        return cls(
            name=d["name"],
            stage_dir=d["stage_dir"],
            seed=int(d["seed"]),
            inputs=dict(d.get("inputs", {})),
            params=dict(d.get("params", {})),
        )

    def write_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=True))

    @classmethod
    def read_yaml(cls, path: str | Path) -> "StageConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))


def cli() -> None:
    """Worker entry point. Resolves the run fn from the registry by --spec.

    Invoked as: `python -m synbio.stages.<module> run --spec <name> --cfg <path>`.
    Importing the module (which __main__ does) registers its specs, so the
    registry lookup below succeeds.
    """
    from synbio.orchestrator import get_run_fn  # local import: avoids cycle
    from synbio.utils.logging import configure_logging
    from synbio.utils.seed import set_seed

    configure_logging()  # worker process: install the INFO stderr handler so stage logs
    #                      (e.g. "finetune step N/2000") reach the captured log.txt — without
    #                      this only WARNING+ leak through Python's lastResort handler.

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--spec", required=True)
    parser.add_argument("--cfg", required=True)
    args = parser.parse_args()

    cfg = StageConfig.read_yaml(args.cfg)
    set_seed(cfg.seed)
    run_fn = get_run_fn(args.spec)
    result = run_fn(cfg)
    result.write_json(Path(cfg.stage_dir) / "result.json")
