"""
Internal logging helper for llm-router-ledger.

Library code calls get_logger(__name__) instead of constructing loggers
directly so that one day we can swap in a different logging backend
without touching every call site. Consumers configure handlers, levels,
and formatting at their application root; this module deliberately does
not.
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger scoped to the given name.

    Pass __name__ from the calling module so log records carry the dotted
    module path, e.g. "llm_router_ledger.dispatcher".
    """
    return logging.getLogger(name)
