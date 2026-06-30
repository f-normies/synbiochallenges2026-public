"""Execution seam: how a stage command is run in its env.

LocalExecutor wraps `synrun <env> <cmd>` as a child process inside the current
srun allocation (the LocalExecutor of the spec). SlurmExecutor is a declared but
unimplemented seam. Tests inject a FakeExecutor (see tests/) that needs no
container — this is why execution is an interface, not inline subprocess code.
"""

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ExecResult",
    "Executor",
    "LocalExecutor",
    "SlurmExecutor",
    "build_synrun_cmd",
]


@dataclass(frozen=True)
class ExecResult:
    """Outcome of running one stage command."""

    returncode: int


def build_synrun_cmd(env: str, cmd: list[str], synrun_bin: str = "synrun") -> list[str]:
    """Build the `synrun <env> <cmd...>` argv."""
    return [synrun_bin, env, *cmd]


class Executor(ABC):
    """Runs a stage command in a named env, teeing output to a log."""

    @abstractmethod
    def run(self, cmd: list[str], env: str, log_path: str | Path) -> ExecResult:
        """Run `cmd` in `env`; write combined stdout/stderr to log_path."""
        raise NotImplementedError


class LocalExecutor(Executor):
    """Run `synrun <env> <cmd>` as a child process in the current allocation."""

    def __init__(self, synrun_bin: str = "synrun", repo_src: str | None = None):
        self.synrun_bin = synrun_bin
        self.repo_src = repo_src  # prepended to PYTHONPATH so workers import synbio

    def run(self, cmd: list[str], env: str, log_path: str | Path) -> ExecResult:
        import os

        full = build_synrun_cmd(env, cmd, self.synrun_bin)
        child_env = dict(os.environ)
        if self.repo_src:
            existing = child_env.get("PYTHONPATH", "")
            child_env["PYTHONPATH"] = (
                f"{self.repo_src}:{existing}" if existing else self.repo_src
            )
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log:
            proc = subprocess.run(
                full, stdout=log, stderr=subprocess.STDOUT, env=child_env, check=False
            )
        return ExecResult(returncode=proc.returncode)


class SlurmExecutor(Executor):
    """Declared seam: one sbatch job per stage. Not implemented in this phase."""

    def run(self, cmd: list[str], env: str, log_path: str | Path) -> ExecResult:
        raise NotImplementedError("SlurmExecutor is a declared seam; use LocalExecutor")
