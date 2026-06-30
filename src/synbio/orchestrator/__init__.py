"""Stage registry, factory, and discovery.

The registry maps a spec name -> (StageSpec, run_fn). It is populated as a side
effect of importing stage modules. `discover_stages()` imports every module in
`synbio.stages` so the orchestrator can build the full DAG from specs alone,
without importing the heavy run() bodies (those lazy-import their deps).
"""

import importlib
import pkgutil
from collections.abc import Callable

from .stage import StageConfig, StageResult, StageSpec

__all__ = [
    "STAGE_REGISTRY",
    "register_stage",
    "StageFactory",
    "get_run_fn",
    "discover_stages",
]

RunFn = Callable[[StageConfig], StageResult]
STAGE_REGISTRY: dict[str, tuple[StageSpec, RunFn]] = {}


def register_stage(spec: StageSpec) -> Callable[[RunFn], RunFn]:
    """Decorator: register a stage's spec and run function under spec.name."""

    def decorator(fn: RunFn) -> RunFn:
        if spec.name in STAGE_REGISTRY:
            raise ValueError(f"stage already registered: {spec.name}")
        STAGE_REGISTRY[spec.name] = (spec, fn)
        return fn

    return decorator


def StageFactory(name: str) -> StageSpec:
    """Return the StageSpec registered under `name`."""
    if name not in STAGE_REGISTRY:
        raise KeyError(f"unknown stage: {name}")
    return STAGE_REGISTRY[name][0]


def get_run_fn(name: str) -> RunFn:
    """Return the run function registered under `name`."""
    if name not in STAGE_REGISTRY:
        raise KeyError(f"unknown stage: {name}")
    return STAGE_REGISTRY[name][1]


def discover_stages() -> None:
    """Import every module under synbio.stages to populate the registry."""
    import synbio.stages as pkg

    for mod in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"synbio.stages.{mod.name}")
