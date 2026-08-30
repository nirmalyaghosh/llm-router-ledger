"""
Pydantic AI model that keeps the ledger.

ledger_model(endpoint_name, tracker=...) builds the model for an
endpoint in llm_endpoints.yaml and wraps it. Every call the agent
makes writes the same paired llm_request and llm_response events
send_message writes:

    from llm_router_ledger.integrations.pydantic_ai import ledger_model

    model = ledger_model("openrouter-mimo-v2.5", tracker=tracker)
    agent = Agent(model)

The alternative is UsageTracker.record_run(), which records after the
fact. Both produce the same rows for a successful run. This one also
records a call that raised, resolves purpose per call, and covers
streaming.

Wraps the built model rather than injecting a client, so the same
wrapper covers non-OpenAI backends. The endpoint's own provider is
stamped on each row; the framework's provider_name is "openai" for
every OpenAI-compatible server.

Requires the pydantic-ai extra:

    uv pip install llm-router-ledger[pydantic-ai]
"""

from __future__ import annotations

import copy

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from llm_router_ledger._logger import get_logger
from llm_router_ledger._run_messages import (
    latest_user_prompt,
    response_text,
    response_usage,
    system_prompt as read_system_prompt,
)
from llm_router_ledger.config import (
    EndpointConfig,
    LLMConfig,
    load_config,
)
from llm_router_ledger.exceptions import (
    ConfigError,
    EndpointNotFoundError,
)
from llm_router_ledger.purpose import current_purpose
from llm_router_ledger.usage_tracker import UsageTracker


try:
    from pydantic_ai.models import (
        Model,
        ModelRequestParameters,
        StreamedResponse,
    )
    from pydantic_ai.models.wrapper import WrapperModel
except ImportError as exc:  # pragma: no cover
    raise ConfigError(
        "Pydantic AI is not installed. Install the optional"
        " extra: uv pip install"
        " llm-router-ledger[pydantic-ai]"
    ) from exc


logger = get_logger(__name__)

__all__ = ["ledger_model"]


def _build_model(
    endpoint: EndpointConfig,
) -> Model:
    """
    Helper function used to build the unwrapped Pydantic AI model for
    one endpoint.

    The SDK client is constructed here rather than left to Pydantic AI's
    environment lookup. api_key_env, base_url, timeout_seconds and
    max_retries then apply as they do to get_client().
    """
    if endpoint.provider == "anthropic":
        try:
            from anthropic import AsyncAnthropic
            from pydantic_ai.models.anthropic import (
                AnthropicModel,
            )
            from pydantic_ai.providers.anthropic import (
                AnthropicProvider,
            )
        except ImportError as exc:
            raise ConfigError(
                "The 'anthropic' SDK is not installed."
                " Install the optional extra: uv pip"
                " install llm-router-ledger[anthropic]"
            ) from exc
        client = AsyncAnthropic(
            api_key=endpoint.api_key,
            timeout=endpoint.timeout_seconds,
            max_retries=endpoint.max_retries,
        )
        return AnthropicModel(
            endpoint.model,
            provider=AnthropicProvider(
                anthropic_client=client,
            ),
        )

    if endpoint.provider == "azure" and not endpoint.base_url:
        raise ConfigError(
            f"Azure endpoint '{endpoint.name}' requires"
            f" base_url (e.g."
            f" https://<resource>.openai.azure.com/openai/v1/)"
        )

    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import (
        OpenAIChatModel,
        OpenAIChatModelSettings,
    )
    from pydantic_ai.providers.openai import OpenAIProvider

    kwargs: dict[str, Any] = {
        "api_key": endpoint.api_key,
        "timeout": endpoint.timeout_seconds,
        "max_retries": endpoint.max_retries,
    }
    if endpoint.base_url:
        kwargs["base_url"] = endpoint.base_url
    settings = None
    if endpoint.extra_body:
        # Deep copied because EndpointConfig is not frozen: the
        # config's dict would otherwise be shared with the model.
        settings = OpenAIChatModelSettings(
            extra_body=copy.deepcopy(endpoint.extra_body),
        )
    return OpenAIChatModel(
        endpoint.model,
        provider=OpenAIProvider(
            openai_client=AsyncOpenAI(**kwargs),
        ),
        settings=settings,
    )


def _partial_usage(
    stream: StreamedResponse,
) -> dict[str, Any] | None:
    """
    Helper function used to read the usage a broken stream produced,
    returning None when it cannot be read.

    Runs on a failure path. Any exception here is logged and discarded
    rather than displacing the provider's own, including a renamed
    pydantic-ai attribute, which would otherwise present silently as an
    empty usage block.
    """
    try:
        return response_usage(stream.get())
    except Exception:
        logger.warning(
            "Could not read the usage off a broken"
            " stream; its tokens are unrecorded",
            exc_info=True,
        )
        return None


class _LedgerModel(WrapperModel):
    """
    Wrapping model that writes a ledger pair around every call.

    Private. ledger_model() is the public API, and the wrapping may
    change without affecting callers.
    """

    def __init__(
        self,
        wrapped: Model,
        *,
        tracker: UsageTracker,
        provider: str,
        purpose: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(wrapped)
        self._tracker = tracker
        # Not _provider: the base Model reserves that name for the
        # framework's own Provider object.
        self._provider_name = provider
        self._purpose = purpose
        self._metadata = metadata

    def _resolve_purpose(self) -> str | None:
        """
        Helper function used to pick the purpose for one call.

        An active purpose_scope takes precedence over the purpose bound at
        construction, which is what lets several agents share one model and
        still land in different rows. With neither set, None defers to the
        tracker's default_purpose.
        """
        return current_purpose() or self._purpose or None

    def _log_failure(
        self,
        request_id: str,
        exc: BaseException,
        usage: dict[str, Any] | None = None,
    ) -> None:
        """
        Helper function used to write the llm_error event for one failed
        call.

        The write is guarded. A tracker that cannot write must not replace
        the provider's exception with its own. Follows UsageTracker._notify.
        """
        try:
            self._tracker.log_error(
                request_id=request_id,
                model=self.model_name,
                error_type=type(exc).__name__,
                error_message=str(exc),
                status_code=getattr(
                    exc,
                    "status_code",
                    None,
                ),
                purpose=self._resolve_purpose(),
                provider=self._provider_name,
                usage=usage,
                metadata=self._metadata,
            )
        except Exception:
            logger.warning(
                "Could not record a failed call; it"
                " leaves an unpaired llm_request",
                exc_info=True,
            )

    def _log_request(self, messages: list[Any]) -> str:
        """
        Helper function used to write the llm_request event for one call and
        return its id.

        Written before the call, as the dispatcher does. A call that raises
        then leaves an unpaired request rather than none.

        The system prompt is read from the whole history. instructions=
        appears on every request; system_prompt= only on the first, which is
        resent as history thereafter.
        """
        prompt = ""
        user = ""
        for message in messages:
            if getattr(message, "kind", None) != "request":
                continue
            prompt = read_system_prompt(message) or prompt
            user = latest_user_prompt(message)
        return self._tracker.record_request(
            model=self.model_name,
            provider=self._provider_name,
            purpose=self._resolve_purpose(),
            system_prompt=prompt,
            user_prompt=user,
            metadata=self._metadata,
        )

    def _log_response(
        self,
        request_id: str,
        response: Any,
    ) -> None:
        """
        Helper function used to write the llm_response event pairing one
        llm_request.

        The provider's reported cost does not reach here: the framework
        keeps only integer usage fields. Reconcile cost against the
        provider's export by response id instead.

        The write is guarded and sits outside the failure handler. A tracker
        that cannot write must not discard a call whose tokens are billed,
        and a fault here belongs to this package rather than the provider.
        """
        try:
            self._log_response_unguarded(request_id, response)
        except Exception:
            logger.warning(
                "Could not record a completed call; it"
                " leaves an unpaired llm_request",
                exc_info=True,
            )

    def _log_response_unguarded(
        self,
        request_id: str,
        response: Any,
    ) -> None:
        """
        Helper function used to write the llm_response event, letting a
        failure propagate.

        Separated from the guard above. This composes the event; the caller
        decides what a write failure means.
        """
        self._tracker.record_response(
            request_id=request_id,
            model=(
                getattr(response, "model_name", "")
                or self.model_name
            ),
            usage=response_usage(response),
            response_id=(
                getattr(
                    response,
                    "provider_response_id",
                    "",
                )
                or ""
            ),
            response_text=response_text(response),
            provider=self._provider_name,
            purpose=self._resolve_purpose(),
            metadata=self._metadata,
        )

    def _log_stream_response(
        self,
        request_id: str,
        stream: StreamedResponse,
    ) -> None:
        """
        Helper function used to write the llm_response event for one
        streamed call.

        get() is called here because these paths may already be unwinding an
        exception, GeneratorExit among them. A failed read is logged and
        discarded, and cannot displace the exception in flight.
        """
        try:
            response = stream.get()
        except Exception:
            logger.warning(
                "Could not read a finished stream; the"
                " call leaves an unpaired llm_request",
                exc_info=True,
            )
            return
        self._log_response(request_id, response)

    async def request(
        self,
        messages: list[Any],
        model_settings: Any,
        model_request_parameters: ModelRequestParameters,
    ) -> Any:
        request_id = self._log_request(messages)
        try:
            response = await self.wrapped.request(
                messages,
                model_settings,
                model_request_parameters,
            )
        except BaseException as exc:
            # BaseException, not Exception: CancelledError derives
            # from it. Cancellation is routine in an agent
            # framework, and catching Exception alone would leave
            # many requests without an outcome.
            self._log_failure(request_id, exc)
            raise
        self._log_response(request_id, response)
        return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[Any],
        model_settings: Any,
        model_request_parameters: ModelRequestParameters,
        run_context: Any = None,
    ) -> AsyncIterator[StreamedResponse]:
        request_id = self._log_request(messages)
        spent: dict[str, Any] | None = None
        finished: StreamedResponse | None = None
        try:
            async with self.wrapped.request_stream(
                messages,
                model_settings,
                model_request_parameters,
                run_context,
            ) as stream:
                try:
                    yield stream
                except GeneratorExit:
                    # Abandoned, not failed: equivalent to breaking
                    # out of the loop, and recorded the same way. An
                    # llm_error would report a failure that did not
                    # occur.
                    self._log_stream_response(request_id, stream)
                    raise
                except BaseException:
                    # Read what the stream produced so the failure
                    # carries the tokens consumed; the handler below
                    # records the event.
                    spent = _partial_usage(stream)
                    raise
                finished = stream
        except GeneratorExit:
            # Recorded as a response above; the failure branch would
            # leave the request carrying both outcomes.
            raise
        except BaseException as exc:
            self._log_failure(request_id, exc, spent)
            raise
        if finished is None:
            # Reachable only when the wrapped context manager
            # suppresses the caller's exception. No shipped model
            # does, but a user's wrapper may. There is no response
            # to record, and get() on None would displace a call
            # that had already returned.
            logger.warning(
                "The wrapped model suppressed a failure;"
                " the call leaves an unpaired"
                " llm_request",
            )
            return
        # Outside the async with. A failure while the context
        # manager closes is then recorded as an error rather than
        # following a response already written. Usage is not final
        # until the stream ends.
        self._log_stream_response(request_id, finished)


def ledger_model(
    endpoint_name: str,
    *,
    tracker: UsageTracker,
    purpose: str = "",
    metadata: dict[str, Any] | None = None,
    config: LLMConfig | None = None,
) -> Model:
    """
    Return a Pydantic AI model for the named endpoint that records every
    call it makes to the tracker.

    The endpoint's provider, model, base_url, api_key_env, extra_body,
    timeout_seconds and max_retries apply. The agent reaches the
    endpoint on the same terms as send_message.

    purpose binds for the life of the model; an active purpose_scope
    overrides it per call. metadata is stamped on every event and never
    reaches the wire. Endpoint request parameters belong in extra_body.

    Raises EndpointNotFoundError if the name is absent from the config,
    and ConfigError if the endpoint requires an SDK that is not
    installed.
    """
    if config is None:
        config = load_config()
    if endpoint_name not in config.endpoints:
        raise EndpointNotFoundError(
            f"Endpoint '{endpoint_name}' not"
            f" found in config"
        )
    endpoint = config.endpoints[endpoint_name]
    return _LedgerModel(
        _build_model(endpoint),
        tracker=tracker,
        provider=endpoint.provider,
        purpose=purpose,
        metadata=metadata,
    )
