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

import json
import os
import uuid

from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

from llm_router_ledger._logger import get_logger
from llm_router_ledger.exceptions import UsageTrackerError


logger = get_logger(__name__)


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
        preview_length: int = 200,
        counter_width: int = 4,
        rotate_daily: bool = False,
        backup_count: int = 30,
    ) -> None:
        """
        Open the JSONL log and start a fresh run.

        The parent directory of log_path is created if it does not exist.
        run_tag and run_label default to the LRL_RUN_TAG and LRL_RUN_LABEL
        environment variables when not passed explicitly.
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
        """
        if len(text) <= self._preview_length:
            return text
        return (
            text[: self._preview_length] + "..."
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
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Write an llm_request event. Returns the request_id (run_id +
        zero-padded counter) so the caller can pair it with the
        corresponding response.
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
            "purpose": (
                purpose
                if purpose is not None
                else self._default_purpose
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
        if metadata:
            entry["metadata"] = metadata
        self._write_entry(entry)
        return request_id

    def log_response(
        self,
        *,
        request_id: str,
        model: str,
        response_text: str,
        usage: dict[str, Any],
        generation_id: str = "",
        purpose: str | None = None,
        usage_details: (
            dict[str, Any] | None
        ) = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Write an llm_response event. The generation_id is auto-routed to
        the "generation_id" key when prefixed "gen-" (OpenRouter
        convention), otherwise to "provider_response_id".
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
            "purpose": (
                purpose
                if purpose is not None
                else self._default_purpose
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
        if usage_details:
            entry["usage_details"] = (
                usage_details
            )
        if metadata:
            entry["metadata"] = metadata
        self._write_entry(entry)

    def start_run(self) -> str:
        """
        Begin a new run: generate a fresh run_id (8-char uuid4 prefix)
        and reset the request counter. Returns the new run_id.
        """
        self._run_id = str(uuid.uuid4())[:8]
        self._counter = 0
        return self._run_id
