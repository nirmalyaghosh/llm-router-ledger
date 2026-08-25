"""
Append-only JSONL usage tracker.

Each LLM call produces exactly two events (llm_request and llm_response)
stamped with project_id, run_tag, run_label, purpose, and a paired
request_id. The tracker never computes cost at runtime; reconciliation
happens by joining the JSONL log on generation_id (or
provider_response_id) against the provider's CSV export.

run_tag and run_label fall back to the LRL_RUN_TAG and LRL_RUN_LABEL
environment variables when the constructor arguments are left as None.
"""

from __future__ import annotations

import copy
import json
import os
import uuid

from collections.abc import (
    Callable,
    Iterable,
)
from datetime import (
    datetime,
    timezone,
)
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

from llm_router_ledger._logger import get_logger
from llm_router_ledger._usage import (
    ORDINARY_FINISH_REASONS,
    map_request_usage,
    normalise_token_keys,
    split_usage,
)
from llm_router_ledger.exceptions import UsageTrackerError
from llm_router_ledger.purpose import current_purpose


logger = get_logger(__name__)


Subscriber = Callable[[dict[str, Any]], None]


def _count_tool_calls(message: Any) -> int:
    """
    Helper function used to count the tool calls in one model response,
    so a turn that returned only tool calls does not read as an empty
    response in the ledger. Mirrors completion_tool_call_count as the
    OpenAI-compatible adapter writes it.

    Counts the ordinary tool call part only. A framework's own
    provider-side call parts, e.g. a hosted web search, carry their own
    discriminators and are not counted, so this is a floor rather than
    an exact total for a run that uses them.
    """
    return sum(
        1
        for part in getattr(message, "parts", ())
        if getattr(part, "part_kind", None) == "tool-call"
    )


def _finish_reason(message: Any) -> str:
    """
    Helper function used to read the finish reason off one model
    response, in the provider's own vocabulary, or "" for a turn that
    ran to completion.

    The provider's word for it is preferred over the framework's
    normalised one, because both adapters record the raw value and a
    reconciler filtering on finish_reason must not have to know that
    the same truncated turn is "tool_calls" from one path and
    "tool_call" from the other. Pydantic AI keeps the raw value in
    provider_details and exposes the normalised one on the message, so
    the raw value is there to be preferred.
    """
    details = getattr(message, "provider_details", None)
    raw = (
        details.get("finish_reason")
        if isinstance(details, dict)
        else None
    )
    reason = raw or getattr(message, "finish_reason", None)
    if not reason or reason in ORDINARY_FINISH_REASONS:
        return ""
    return str(reason)


def _latest_user_prompt(message: Any) -> str:
    """
    Helper function used to pull the last user prompt out of one model
    request, for the request event's preview and character count. Tool
    returns and system prompts are skipped: the dispatcher previews the
    latest user message, and this follows it.

    Returns "" for a request carrying no user prompt, e.g. the tool
    return that continues a tool loop, and for a multimodal prompt
    whose content is not a plain string.
    """
    latest = ""
    for part in getattr(message, "parts", ()):
        if getattr(part, "part_kind", None) != "user-prompt":
            continue
        content = getattr(part, "content", "")
        if isinstance(content, str):
            latest = content
    return latest


def _response_text(message: Any) -> str:
    """
    Helper function used to join the text parts of one model response,
    for the response event's preview and character count.

    Thinking parts are excluded: they are not the answer, and their
    tokens are already accounted for under
    completion_reasoning_tokens. A turn that returned only tool calls
    yields "", the same as the adapter records for it.
    """
    return "".join(
        part.content
        for part in getattr(message, "parts", ())
        if getattr(part, "part_kind", None) == "text"
        and isinstance(getattr(part, "content", None), str)
    )


class UsageTracker:
    """
    Append paired llm_request and llm_response events to a JSONL log.

    State is fully encapsulated; instantiate one tracker per run, or
    share across runs by calling start_run() to mint a new run_id and
    reset the counter.
    """

    def __init__(
        self,
        *,
        log_path: str | Path,
        project_id: str,
        run_tag: str | None = None,
        run_label: str | None = None,
        default_purpose: str = "",
        preview_length: int = 0,
        counter_width: int = 4,
        rotate_daily: bool = False,
        backup_count: int = 30,
    ) -> None:
        """
        Open the JSONL log and start a fresh run.

        The parent directory of log_path is created if it does not exist.
        run_tag and run_label default to the LRL_RUN_TAG and LRL_RUN_LABEL
        environment variables when not passed explicitly.

        preview_length defaults to 0: prompt and response previews are
        redacted (written as "[REDACTED]" when the underlying text is
        non-empty, "" when it genuinely is empty) so no call content
        reaches the ledger unless a caller opts in. Pass a positive value
        to store up to that many characters of each prompt and response,
        truncated with a trailing "..." when the text is longer.
        """
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._project_id = project_id
        self._run_tag = (
            run_tag
            if run_tag is not None
            else os.environ.get(
                "LRL_RUN_TAG",
                "",
            )
        )
        self._run_label = (
            run_label
            if run_label is not None
            else os.environ.get(
                "LRL_RUN_LABEL",
                "",
            )
        )
        self._default_purpose = default_purpose
        self._preview_length = preview_length
        self._counter_width = counter_width
        self._rotate_daily = rotate_daily
        self._backup_count = backup_count
        self._handler: (
            TimedRotatingFileHandler | None
        ) = None
        self._stream: Any = None
        self._run_id: str = ""
        self._counter: int = 0
        self._subscribers: list[Subscriber] = []
        self._open_stream()
        self.start_run()

    def __enter__(self) -> UsageTracker:
        return self

    @property
    def run_id(self) -> str:
        """
        The current run identifier (8-char uuid4 prefix). Stamped as the
        prefix of every request_id emitted by this tracker; consumers can
        read it to tag related records in sibling log files.
        """
        return self._run_id

    def __exit__(
        self,
        exc_type: Any,
        exc: Any,
        tb: Any,
    ) -> None:
        self.close()

    def _make_preview(self, text: str) -> str:
        """
        Helper function used to truncate a long prompt or response to
        preview_length chars, suffixed with "..." when truncation actually
        happened.

        preview_length <= 0 means previews are redacted: non-empty text
        becomes "[REDACTED]" rather than "", so a reader can tell content
        was withheld apart from content that was never there (e.g. an
        embedding response, or no system prompt).
        """
        if self._preview_length <= 0:
            return "[REDACTED]" if text else ""
        if len(text) <= self._preview_length:
            return text
        return (
            text[: self._preview_length] + "..."
        )

    def _notify(self, entry: dict[str, Any]) -> None:
        """
        Helper function used to hand a finished entry to every subscriber
        after it is safely on disk. A subscriber that raises is logged and
        skipped, never propagated: an unreachable sink or a broken
        callback must not fail the LLM call that produced the entry, and
        must not stop the remaining subscribers from running.

        Each subscriber gets its own deep copy, so one that mutates the
        dict cannot corrupt what the next one sees.
        """
        for subscriber in self._subscribers:
            try:
                subscriber(copy.deepcopy(entry))
            except Exception:
                logger.exception(
                    "Usage subscriber %(subscriber)r raised on"
                    " event %(event)s; the entry is already"
                    " in the ledger and the error is"
                    " being ignored",
                    {
                        "subscriber": getattr(
                            subscriber,
                            "__name__",
                            subscriber,
                        ),
                        "event": entry.get("event", "?"),
                    },
                )

    def _open_stream(self) -> None:
        """
        Helper function used to open the underlying write stream once,
        either as a plain append file or via TimedRotatingFileHandler
        when rotate_daily is set.
        """
        if self._rotate_daily:
            self._handler = (
                TimedRotatingFileHandler(
                    filename=str(
                        self._log_path,
                    ),
                    when="midnight",
                    interval=1,
                    backupCount=(
                        self._backup_count
                    ),
                    encoding="utf-8",
                )
            )
            self._stream = self._handler.stream
        else:
            self._stream = open(
                self._log_path,
                "a",
                encoding="utf-8",
            )

    def _resolve_purpose(
        self,
        purpose: str | None,
    ) -> str:
        """
        Helper function used to pick the purpose for one entry.

        A purpose passed to the call wins; failing that the ambient
        purpose_scope, if one is active; failing that the tracker's
        default_purpose. The narrowest thing that was actually set is
        what reaches the ledger.

        An empty string counts as unset rather than as an explicit
        override, because send_message defaults its purpose argument to
        "" and passes it through on every call. Treating it as an
        override would mean no call routed through the dispatcher could
        ever see the scope or the default.
        """
        if purpose:
            return purpose
        scoped = current_purpose()
        if scoped:
            return scoped
        return self._default_purpose

    def _write_entry(self, entry: dict[str, Any]) -> None:
        """
        Helper function used to serialise a single dict to JSON and
        append a terminated line to the log. Raises UsageTrackerError if
        the underlying stream has already been closed.
        """
        if self._stream is None:
            raise UsageTrackerError(
                "Tracker is closed;"
                " cannot write further entries"
            )
        line = json.dumps(entry, default=str)
        self._stream.write(line + "\n")
        self._stream.flush()
        self._notify(entry)

    def close(self) -> None:
        """
        Flush and release the underlying file handle. Safe to call more
        than once.
        """
        if self._handler is not None:
            try:
                self._handler.close()
            finally:
                self._handler = None
                self._stream = None
            return
        if self._stream is not None:
            try:
                self._stream.flush()
                self._stream.close()
            finally:
                self._stream = None

    def log_request(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        purpose: str | None = None,
        provider: str = "",
        modality: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Write an llm_request event. Returns the request_id (run_id +
        zero-padded counter) so the caller can pair it with the
        corresponding response.

        provider is the EndpointConfig.provider value (e.g. "ollama",
        "openrouter", "azure"). Recorded verbatim so consumers can group
        ledger entries by which server produced the tokens.

        modality names the capability the call used, e.g. "embedding".
        The key is written only when it is not "text", so text entries
        keep the exact shape they had before the field existed and a
        reader may treat an absent modality as text.
        """
        self._counter += 1
        width = self._counter_width
        request_id = (
            f"{self._run_id}"
            f"-{self._counter:0{width}d}"
        )
        entry: dict[str, Any] = {
            "event": "llm_request",
            "project_id": self._project_id,
            "purpose": self._resolve_purpose(
                purpose,
            ),
            "request_id": request_id,
            "run_tag": self._run_tag,
            "run_label": self._run_label,
            "timestamp": (
                datetime.now(timezone.utc)
                .isoformat()
            ),
            "model": model,
            "system_prompt_preview": (
                self._make_preview(system_prompt)
            ),
            "user_prompt_preview": (
                self._make_preview(user_prompt)
            ),
            "user_prompt_length": len(
                user_prompt,
            ),
        }
        if provider:
            entry["provider"] = provider
        if modality != "text":
            entry["modality"] = modality
        if metadata:
            entry["metadata"] = metadata
        self._write_entry(entry)
        return request_id

    def log_error(
        self,
        *,
        request_id: str,
        model: str,
        error_type: str,
        error_message: str,
        status_code: int | None = None,
        purpose: str | None = None,
        provider: str = "",
        modality: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Write an llm_error event for a call that raised.

        Shares request_id with the llm_request that preceded it, so a
        failure pairs the same way a success does and no request is left
        orphaned. A third event type rather than an llm_response with an
        error field: a failed call produced no tokens, and writing zeroes
        into usage would corrupt anyone summing them.

        error_type is the original exception's class name, kept because
        the wrapped class the caller sees is coarser than what the SDK
        raised. status_code is the provider's HTTP status where there was
        one, and is omitted for transport failures.

        The SDK retries internally before raising, so one llm_error
        stands for however many attempts it made.
        """
        entry: dict[str, Any] = {
            "event": "llm_error",
            "project_id": self._project_id,
            "purpose": self._resolve_purpose(
                purpose,
            ),
            "request_id": request_id,
            "run_tag": self._run_tag,
            "run_label": self._run_label,
            "timestamp": (
                datetime.now(timezone.utc)
                .isoformat()
            ),
            "model": model,
            "error_type": error_type,
            "error_message": error_message,
        }
        if status_code is not None:
            entry["status_code"] = status_code
        if provider:
            entry["provider"] = provider
        if modality != "text":
            entry["modality"] = modality
        if metadata:
            entry["metadata"] = metadata
        self._write_entry(entry)

    def log_response(
        self,
        *,
        request_id: str,
        model: str,
        response_text: str,
        usage: dict[str, Any],
        generation_id: str = "",
        purpose: str | None = None,
        provider: str = "",
        modality: str = "text",
        usage_details: (
            dict[str, Any] | None
        ) = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Write an llm_response event. The generation_id is auto-routed to
        the "generation_id" key when prefixed "gen-" (OpenRouter
        convention), otherwise to "provider_response_id".

        provider mirrors the value passed to the paired log_request call
        and identifies the server that produced the tokens.

        modality mirrors the paired log_request call and is written only
        when it is not "text", leaving existing text entries unchanged.
        """
        id_key = (
            "generation_id"
            if generation_id.startswith("gen-")
            else "provider_response_id"
        )
        prompt_tokens = (
            usage.get("prompt_tokens")
            or usage.get("input_tokens", 0)
        )
        completion_tokens = (
            usage.get("completion_tokens")
            or usage.get("output_tokens", 0)
        )
        total_tokens = (
            usage.get("total_tokens")
            or (
                usage.get("input_tokens", 0)
                + usage.get("output_tokens", 0)
            )
        )
        entry: dict[str, Any] = {
            "event": "llm_response",
            "project_id": self._project_id,
            "purpose": self._resolve_purpose(
                purpose,
            ),
            "request_id": request_id,
            "run_tag": self._run_tag,
            "run_label": self._run_label,
            id_key: generation_id,
            "timestamp": (
                datetime.now(timezone.utc)
                .isoformat()
            ),
            "model": model,
            "response_preview": (
                self._make_preview(response_text)
            ),
            "response_length": len(
                response_text,
            ),
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": (
                    completion_tokens
                ),
                "total_tokens": total_tokens,
            },
        }
        if provider:
            entry["provider"] = provider
        if modality != "text":
            entry["modality"] = modality
        if usage_details:
            entry["usage_details"] = (
                usage_details
            )
        if metadata:
            entry["metadata"] = metadata
        self._write_entry(entry)

    def record_request(
        self,
        *,
        model: str,
        provider: str = "",
        purpose: str | None = None,
        system_prompt: str = "",
        user_prompt: str = "",
        modality: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Write an llm_request event for a call this library did not make
        itself, and return the request_id to pair a response with.

        Use this when an agent framework, a raw SDK call, or anything
        else owns the call path and the ledger still has to account for
        the tokens. The event is the one log_request writes, so a
        recorded call and a send_message call are the same row shape
        and reconcile the same way.

        system_prompt and user_prompt are optional, since a caller
        recording someone else's call may not have them. Leaving them
        empty records an empty preview and a zero length, which reads
        as "no text was supplied" rather than as an empty prompt.

        A call that fails leaves this request unpaired, exactly as it
        does for send_message.
        """
        return self.log_request(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            purpose=purpose,
            provider=provider,
            modality=modality,
            metadata=metadata,
        )

    def record_response(
        self,
        *,
        request_id: str,
        model: str,
        usage: dict[str, Any] | None = None,
        response_id: str = "",
        response_text: str = "",
        provider: str = "",
        purpose: str | None = None,
        modality: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Write the llm_response event pairing a record_request call.

        usage is the provider's usage mapping as reported, not a
        pre-split one: the three token keys are lifted into the usage
        block and everything else is written under usage_details, the
        same split send_message performs. Pass None for a call that
        reported no usage at all; the token keys are then written as
        zero, which is what an absent usage block already produces.

        A mapping reporting input_tokens and output_tokens, as
        Anthropic does, is accepted as readily as one reporting
        prompt_tokens and completion_tokens, and total_tokens is filled
        in when the provider omitted it.

        response_id is routed the way log_response routes it: an id
        prefixed "gen-" lands in generation_id, anything else in
        provider_response_id.
        """
        token_usage, usage_details = split_usage(
            normalise_token_keys(usage or {}),
        )
        self.log_response(
            request_id=request_id,
            model=model,
            response_text=response_text,
            usage=token_usage,
            generation_id=response_id,
            purpose=purpose,
            provider=provider,
            modality=modality,
            usage_details=usage_details,
            metadata=metadata,
        )

    def record_run(
        self,
        messages: Iterable[Any],
        *,
        purpose: str | None = None,
        provider: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """
        Record every model call in a finished Pydantic AI run, and
        return the request_ids written, in order.

        Pass the run's message list:

            result = await agent.run("...")
            tracker.record_run(
                result.all_messages(),
                purpose="query-planning",
            )

        Each model response in the list carries its own usage, model
        name and response id, so one request and response pair is
        written per model call. A run that called a tool three times
        produces three pairs, not one.

        The messages are read by duck typing rather than by importing
        pydantic-ai, so recording a run costs the core package no
        dependency.

        provider overrides the provider name stamped on the rows. Pass
        the endpoint's provider from llm_endpoints.yaml: the name on
        the message is the framework's own provider id, which is
        "openai" for every OpenAI-compatible server, so an unoverridden
        row cannot tell an OpenAI call from a local one.

        Two limits are worth knowing before reconciling against these
        rows:

        - A finished message list carries no record of why each call
          was made, so one purpose is stamped across the whole run and
          retries within it inherit it.
        - A run that raises produces no rows at all, unlike
          send_message, which logs the request before making the call.
          Both are correct, but it changes what an unpaired llm_request
          means: from this method it means the process died mid-write,
          not that a call failed.
        - A provider's own reported cost does not survive the trip. The
          framework keeps only integer usage fields, so a reported cost
          never reaches these rows and usage_details carries the
          locally computed estimated_cost instead. Reconcile these rows
          against the provider's export by response id, which is the
          documented method regardless.
        """
        request_ids: list[str] = []
        user_prompt = ""
        for message in messages:
            kind = getattr(message, "kind", None)
            if kind == "request":
                user_prompt = _latest_user_prompt(message)
                continue
            if kind != "response":
                continue
            model = getattr(message, "model_name", "") or ""
            stamped_provider = (
                provider
                or getattr(message, "provider_name", "")
                or ""
            )
            request_id = self.record_request(
                model=model,
                provider=stamped_provider,
                purpose=purpose,
                user_prompt=user_prompt,
                metadata=metadata,
            )
            usage = map_request_usage(
                getattr(message, "usage", None),
            )
            finish_reason = _finish_reason(message)
            if finish_reason:
                usage["finish_reason"] = finish_reason
            tool_calls = _count_tool_calls(message)
            if tool_calls:
                usage["completion_tool_call_count"] = (
                    tool_calls
                )
            self.record_response(
                request_id=request_id,
                model=model,
                usage=usage,
                response_id=(
                    getattr(
                        message,
                        "provider_response_id",
                        "",
                    )
                    or ""
                ),
                response_text=_response_text(message),
                provider=stamped_provider,
                purpose=purpose,
                metadata=metadata,
            )
            request_ids.append(request_id)
            user_prompt = ""
        return request_ids

    def start_run(self) -> str:
        """
        Begin a new run: generate a fresh run_id (8-char uuid4 prefix)
        and reset the request counter. Returns the new run_id.
        """
        self._run_id = str(uuid.uuid4())[:8]
        self._counter = 0
        return self._run_id

    def subscribe(self, callback: Subscriber) -> None:
        """
        Register a callback invoked with every entry once it has been
        written to the JSONL ledger. Use this to mirror usage into another
        store (a database, a queue) without the library taking on a
        dependency on that store.

        The callback receives the entry dict, the same shape as the JSONL
        line; its return value is ignored. Callbacks run synchronously on
        the calling thread, so a slow one delays every LLM call. Remote
        destinations should queue the work inside the callback.

        Each entry reaches the ledger before any subscriber runs, and a
        callback that raises is logged and skipped, so a subscriber can
        neither suppress a ledger entry nor fail the call that produced
        it.
        """
        self._subscribers.append(callback)
