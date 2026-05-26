"""
Module-mode entry point. Enables `python -m llm_router_ledger ...` as an
alternative to the installed `llm-router-ledger` script.
"""

import sys

from llm_router_ledger.cli import main


if __name__ == "__main__":
    sys.exit(main())
