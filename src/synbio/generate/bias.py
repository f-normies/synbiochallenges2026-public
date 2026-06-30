"""TGP-style surface-recharge prior as a LigandMPNN per-residue AA bias."""

__all__ = ["recharge_bias"]


def recharge_bias(
    exposed_positions: list[int], b: float, chain: str = "A"
) -> dict[str, dict[str, float]]:
    """Per-residue bias toward E/D/Q and away from K/R on exposed positions."""
    weights = {"E": b, "D": b, "Q": b / 2, "K": -b, "R": -b}
    return {f"{chain}{p}": dict(weights) for p in exposed_positions}
