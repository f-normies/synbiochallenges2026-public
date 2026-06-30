"""Module-level logger factory with a single shared handler."""

import logging
import sys

__all__ = ["get_logger", "configure_logging"]

_CONFIGURED = False
_FMT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Install a single stderr handler on the root 'synbio' logger once."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FMT))
    root = logging.getLogger("synbio")
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring the shared handler on first use."""
    configure_logging()
    return logging.getLogger(name)
