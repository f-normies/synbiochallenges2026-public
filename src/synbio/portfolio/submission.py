from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = ["write_submission"]


def write_submission(selected: pd.DataFrame, path: str | Path, team_name: str = "DOGMA") -> None:
    """Write the competition submission.csv (UTF-8 no BOM, LF, AA only)."""
    ordered = selected.sort_values("seq_id")
    lines = ["Team_Name,Seq_ID,Sequence"]
    for _, row in ordered.iterrows():
        lines.append(f"{team_name},{int(row['seq_id'])},{str(row['sequence'])}")
    text = "\n".join(lines) + "\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")
