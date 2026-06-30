"""Build the stability training data from megascale.csv (paper A.1.4.4).

Melts the pair-shaped table into a per-sequence (sequence -> absolute dG_ML) table for
ridge fitting, plus a pair table for ΔΔG-by-subtraction evaluation. Also builds the GFP
reversion panel (sfGFP -> avGFP single substitutions) for the non-blocking transfer sanity.

dG_ML is folding stability (higher = more stable). The probe trains on dG_ML and emits
ΔΔG_funnel = dG_pred(wt) - dG_pred(mut) (positive = destabilizing). See the design spec.
"""

import pandas as pd

__all__ = ["build_stability_dataset", "build_gfp_reversion_panel"]


def build_stability_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (seq_table, pairs).

    seq_table: one row per unique (WT_name, sequence) with its absolute dG and split —
        the union of every mutant endpoint (mut_seq, dG_ML_mut) and every background
        endpoint (wt_seq, dG_ML_wt). This is the per-sequence ridge training table.
    pairs: (wt_seq, mut_seq, ddG_ML, WT_name, split) — for ΔΔG-by-subtraction evaluation.
    """
    mut = pd.DataFrame(
        {
            "sequence": df["mut_seq"],
            "dG": df["dG_ML_mut"].astype(float),
            "split": df["split"],
            "WT_name": df["WT_name"],
        }
    )
    wt = pd.DataFrame(
        {
            "sequence": df["wt_seq"],
            "dG": df["dG_ML_wt"].astype(float),
            "split": df["split"],
            "WT_name": df["WT_name"],
        }
    )
    seq_table = (
        pd.concat([mut, wt], ignore_index=True)
        .drop_duplicates(subset=["WT_name", "sequence"], keep="first")
        .reset_index(drop=True)
    )
    pairs = pd.DataFrame(
        {
            "wt_seq": df["wt_seq"],
            "mut_seq": df["mut_seq"],
            "ddG_ML": df["ddG_ML"].astype(float),
            "WT_name": df["WT_name"],
            "split": df["split"],
        }
    ).reset_index(drop=True)
    return seq_table, pairs


def build_gfp_reversion_panel(sfgfp_seq: str, avgfp_seq: str) -> list[str]:
    """Single sfGFP->avGFP reversions: sfGFP with each differing position set to avGFP's residue.

    Reverting a superfolder-class change toward avGFP should be destabilizing in aggregate
    (sfGFP is the more-stable superfolder). Requires equal length (1:1 alignment, no indels).
    """
    if len(sfgfp_seq) != len(avgfp_seq):
        raise ValueError(
            f"length mismatch: sfGFP {len(sfgfp_seq)} vs avGFP {len(avgfp_seq)}"
        )
    panel: list[str] = []
    for i, (a, b) in enumerate(zip(sfgfp_seq, avgfp_seq)):
        if a != b:
            panel.append(sfgfp_seq[:i] + b + sfgfp_seq[i + 1 :])
    return panel
