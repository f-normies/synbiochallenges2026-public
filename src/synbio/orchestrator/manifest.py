"""RunManifest: the deterministic decision tree rendered for the PDF.

One StageNode per executed DAG node, with hashes, counts, and recorded decisions.
Renders to JSON and to a human-readable text tree.
"""

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .stage import Decision

__all__ = [
    "StageNode",
    "RunManifest",
    "hash_path",
    "env_snapshot",
]


def hash_path(path: str | Path) -> str:
    """SHA256 of a file, or of a directory's sorted (relpath, content) stream."""
    p = Path(path)
    h = hashlib.sha256()
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file():
                h.update(str(f.relative_to(p)).encode())
                h.update(f.read_bytes())
    else:
        h.update(p.read_bytes())
    return h.hexdigest()


def env_snapshot() -> dict[str, str]:
    """Capture a minimal reproducibility snapshot (extended by workers if needed)."""
    snap = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        snap["torch_version"] = torch.__version__
        snap["cuda_version"] = str(torch.version.cuda)
    except ImportError:
        pass
    return snap


@dataclass
class StageNode:
    """One executed stage in the manifest."""

    stage: str
    env: str
    status: str
    cmd: str
    seed: int
    duration_s: float
    input_hashes: dict[str, str]
    output_hashes: dict[str, str]
    outputs: dict[str, str]  # artifact key -> path recorded for downstream resolution
    n_in: int | None = None
    n_out: int | None = None
    decisions: list[Decision] = field(default_factory=list)
    agent_decision_record: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["decisions"] = [asdict(x) for x in self.decisions]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StageNode":
        d = dict(d)
        d["decisions"] = [Decision(**x) for x in d.get("decisions", [])]
        return cls(**d)


@dataclass
class RunManifest:
    """Root manifest for one pipeline run."""

    run_id: str
    git_sha: str
    config_hash: str
    seed: int
    env_snapshot: dict[str, str]
    nodes: list[StageNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "git_sha": self.git_sha,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "env_snapshot": self.env_snapshot,
            "nodes": [n.to_dict() for n in self.nodes],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunManifest":
        return cls(
            run_id=d["run_id"],
            git_sha=d["git_sha"],
            config_hash=d["config_hash"],
            seed=int(d["seed"]),
            env_snapshot=dict(d.get("env_snapshot", {})),
            nodes=[StageNode.from_dict(n) for n in d.get("nodes", [])],
        )

    def write_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def read_json(cls, path: str | Path) -> "RunManifest":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def node(self, stage: str) -> "StageNode | None":
        """Return the recorded node for `stage`, or None."""
        for n in self.nodes:
            if n.stage == stage:
                return n
        return None

    def render_tree(self) -> str:
        """Render the manifest as an indented text decision tree."""
        lines = [f"run {self.run_id} (git {self.git_sha}, cfg {self.config_hash})"]
        for n in self.nodes:
            flow = ""
            if n.n_in is not None or n.n_out is not None:
                flow = f"  [{n.n_in} -> {n.n_out}]"
            lines.append(f"├─ {n.stage} ({n.env}) {n.status} {n.duration_s:.1f}s{flow}")
            for d in n.decisions:
                kd = f"kept={d.kept} dropped={d.dropped}"
                lines.append(f"│    • {d.name} [{d.threshold}] {kd} {d.note}".rstrip())
        return "\n".join(lines)
