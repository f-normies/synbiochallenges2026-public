"""Stage 08 portfolio selection: pure logic (CPU)."""

from .eligibility import annotate_eligibility, load_exclusion_set
from .selection import SelectionResult, select_portfolio
from .submission import write_submission

__all__ = [
    "annotate_eligibility",
    "load_exclusion_set",
    "SelectionResult",
    "select_portfolio",
    "write_submission",
]
