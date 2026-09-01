"""
Unit tests for llm_router_ledger.results.

ChatResult and EmbeddingResult are frozen dataclasses. They were
NamedTuple subclasses through 0.2.x, where positional unpacking
worked and logged a deprecation warning.
"""

from __future__ import annotations

import dataclasses

import pytest

from llm_router_ledger.results import (
    ChatResult,
    EmbeddingResult,
)


@pytest.mark.parametrize(
    "result",
    [
        ChatResult(
            text="hello",
            usage={"prompt_tokens": 1},
            generation_id="gen-1",
        ),
        EmbeddingResult(
            vectors=[[0.1]],
            usage={"prompt_tokens": 1},
            generation_id="",
        ),
    ],
    ids=["chat", "embedding"],
)
def test_result_is_frozen_and_not_a_sequence(result: object) -> None:
    """
    Unpacking, indexing, iteration and field assignment raise. A
    result does not compare equal to a tuple of its own field
    values, and attribute access returns the field.
    """
    with pytest.raises(TypeError):
        _a, _b, _c = result  # type: ignore[misc]
    with pytest.raises(TypeError):
        result[0]  # type: ignore[index]
    with pytest.raises(TypeError):
        list(result)  # type: ignore[call-overload]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.usage = {}  # type: ignore[misc]
    assert result != dataclasses.astuple(result)
    assert result.usage == {"prompt_tokens": 1}
