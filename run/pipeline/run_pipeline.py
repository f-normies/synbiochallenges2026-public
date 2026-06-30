"""Hydra entry point: compose config -> plain dict -> orchestrator.runner."""

import hydra
from omegaconf import DictConfig, OmegaConf

from synbio.orchestrator.runner import run_pipeline


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    raw = OmegaConf.to_container(cfg, resolve=True)
    config = {
        "run_id": raw["run_id"],
        "runs_dir": raw["runs_dir"],
        "seed": raw["seed"],
        "git_sha": raw.get("git_sha", ""),
        "artifacts": raw["artifacts"],
        "stages": raw["stages"],  # bare list of stage names
        "params": raw.get("params", {}).get("params", {}),  # params group -> dict
    }
    manifest = run_pipeline(config, force=bool(raw.get("force", False)))
    print(manifest.render_tree())


if __name__ == "__main__":
    main()
