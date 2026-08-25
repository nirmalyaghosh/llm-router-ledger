"""
Ambient purpose for calls the caller cannot pass a purpose to.

send_message takes purpose per call, but an agent framework owns the
call path: by the time a request reaches the ledger there is no
argument left to carry it. The usual shape with those frameworks is one
model or one client shared by several agents, so binding purpose at
construction is not enough either.

So purpose is also readable from a context variable, set around the
work rather than passed through it:

    with purpose_scope("query-planning"):
        result = await agent.run("...")

Being a contextvar rather than a module global, it is per-task and per
-thread: two agents running concurrently under asyncio each see their
own value, and neither sees the other's.

A purpose passed explicitly always wins over the scope. The scope in
turn wins over the tracker's default_purpose, so the narrowest thing
that was actually set is what reaches the ledger.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar


_PURPOSE: ContextVar[str] = ContextVar(
    "llm_router_ledger_purpose",
    default="",
)


def current_purpose() -> str:
    """
    Return the purpose in effect for the current context, or "" when no
    scope is active.

    Mostly of interest to integrations, which read it at request time.
    """
    return _PURPOSE.get()


@contextmanager
def purpose_scope(purpose: str) -> Generator[None, None, None]:
    """
    Set the ambient purpose for the duration of the block.

    Scopes nest, and the innermost one wins. The previous value is
    restored on exit, including when the block raises.

    Passing "" enters a scope that clears an outer one, which is the
    way to make a nested call record with no purpose rather than
    inheriting the one around it.
    """
    token = _PURPOSE.set(purpose)
    try:
        yield
    finally:
        _PURPOSE.reset(token)
