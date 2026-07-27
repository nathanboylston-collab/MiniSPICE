"""Browser-based development UI for MiniSPICE.

Optional: requires the ``web`` extra (``pip install -e ".[web]"``).
Run with ``python -m minispice.web``.
"""

from .app import create_app

__all__ = ["create_app"]
