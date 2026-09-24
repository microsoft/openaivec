import asyncio
import json
from collections import Counter
from dataclasses import dataclass, field
from logging import Logger, getLogger
from typing import Any, Generic, cast

import tiktoken
from openai import AsyncOpenAI, OpenAI
from openai.types.responses import ParsedResponse
from openai.types.responses import Response as OAIResponse
from openai.types.responses.response_input_param import ResponseInputParam
from pydantic import BaseModel, ValidationError

from openaivec._cache import AsyncBatchCache, BatchCache
from openaivec._cache.proxy import DEFAULT_MANAGED_CACHE_SIZE
from openaivec._log import observe
from openaivec._model import PreparedTask, ResponseFormat
from openaivec._multimodal import (
    AsyncMultimodalContentBuilder,
    MultimodalContentBuilder,
    is_multimodal_input,
    is_readable_text_file,
    local_file_cache_key,
    read_local_file_for_cache_key,
    read_text_file,
)
from openaivec._retry import RetryPolicy, call_with_retry, call_with_retry_async, retry_deadline

__all__ = [
    "AsyncBatchResponses",
    "BatchResponses",
    "ResponseLimits",
]

_LOGGER: Logger = getLogger(__name__)
_MAX_VALIDATION_FEEDBACK_ITEMS = 8
_LITERAL_INPUT_PREFIX = "\0text:"


@dataclass(frozen=True)
class ResponseLimits:
    """Estimated per-request context budget for batched text Responses.

    ``max_request_tokens`` is the model/deployment context limit, not an API
    parameter. Override it and ``encoding_name`` for custom Azure deployments.
    Output and validation allowances are estimates; choose values appropriate
    for the task and model. A correction whose actual feedback exceeds the
    configured budget fails before a request is sent. Multimodal requests use
    a separate path.
    """

    max_request_tokens: int = 128000
    max_inputs: int = 128
    expected_output_tokens_per_item: int = 256
    validation_feedback_tokens: int = 256
    encoding_name: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_request_tokens",
            "max_inputs",
            "expected_output_tokens_per_item",
            "validation_feedback_tokens",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < (0 if name == "validation_feedback_tokens" else 1):
                raise ValueError(f"{name} must be {'>= 0' if name == 'validation_feedback_tokens' else '> 0'}")


def _plan_response_batches(
    messages: list["Message[str]"],
    model_name: str,
    instructions: str,
    response_format: type,
    limits: ResponseLimits,
    max_validation_retries: int,
    api_kwargs: dict[str, Any],
) -> list[list["Message[str]"]]:
    if limits.encoding_name is not None:
        encoding = tiktoken.get_encoding(limits.encoding_name)
    else:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("o200k_base")

    class MessageT(BaseModel):
        id: int
        body: response_format  # type: ignore

    class ResponseT(BaseModel):
        assistant_messages: list[MessageT]

    def count(text: str) -> int:
        return len(encoding.encode_ordinary(text))

    schema_tokens = count(json.dumps(ResponseT.model_json_schema(), separators=(",", ":")))
    fixed_tokens = count(instructions) + schema_tokens
    if max_validation_retries:
        fixed_tokens += limits.validation_feedback_tokens
    # Allow for request framing and message roles not included in the JSON envelope.
    fixed_tokens += 32
    output_cap = api_kwargs.get("max_output_tokens", 0)
    if (
        not isinstance(output_cap, int)
        or isinstance(output_cap, bool)
        or ("max_output_tokens" in api_kwargs and output_cap < 1)
    ):
        raise ValueError("max_output_tokens must be a positive integer")

    def cost(batch: list[Message[str]]) -> int:
        envelope = Request(user_messages=batch).model_dump(mode="json")
        input_tokens = count(json.dumps(envelope, ensure_ascii=False, separators=(",", ":")))
        output_tokens = max(output_cap, limits.expected_output_tokens_per_item * len(batch))
        return fixed_tokens + input_tokens + output_tokens

    batches: list[list[Message[str]]] = []
    batch: list[Message[str]] = []
    for message in messages:
        candidate = [*batch, message]
        if batch and (len(candidate) > limits.max_inputs or cost(candidate) > limits.max_request_tokens):
            batches.append(batch)
            batch = []
            candidate = [message]
        required = cost(candidate)
        if required > limits.max_request_tokens:
            raise ValueError(
                f"Response input ID {message.id} exceeds the {limits.max_request_tokens}-token request budget "
                f"(estimated {required} tokens including instructions, schema and output)"
            )
        batch = candidate
    if batch:
        batches.append(batch)
    return batches


def _ensure_retry_fits_budget(
    messages: list["Message[str]"],
    model_name: str,
    instructions: str,
    response_format: type,
    limits: ResponseLimits,
    api_kwargs: dict[str, Any],
) -> None:
    planned = _plan_response_batches(messages, model_name, instructions, response_format, limits, 0, api_kwargs)
    if len(planned) != 1:
        raise ValueError("Validation feedback exceeds the configured response token budget")


def _validate_response_ids(expected_ids: list[int], response_ids: list[int]) -> None:
    expected = set(expected_ids)
    received = set(response_ids)
    if len(response_ids) == len(expected_ids) and received == expected:
        return
    duplicates = [identity for identity, count in Counter(response_ids).items() if count > 1]
    message = (
        "Response IDs must match every input ID exactly once. "
        f"Missing IDs: {sorted(expected - received)[:_MAX_VALIDATION_FEEDBACK_ITEMS]}; "
        f"unknown IDs: {sorted(received - expected)[:_MAX_VALIDATION_FEEDBACK_ITEMS]}; "
        f"duplicate IDs: {duplicates[:_MAX_VALIDATION_FEEDBACK_ITEMS]}."
    )
    raise ValidationError.from_exception_data(
        "Response",
        [
            {
                "type": "value_error",
                "loc": ("assistant_messages",),
                "input": response_ids,
                "ctx": {"error": ValueError(message)},
            }
        ],
    )


def _format_validation_error_location(loc: tuple[Any, ...]) -> str:
    """Format a Pydantic validation location tuple into a readable path."""
    path = ""
    for part in loc:
        if isinstance(part, int):
            path = f"{path}[{part}]" if path else f"[{part}]"
        else:
            token = str(part)
            path = f"{path}.{token}" if path else token
    return path or "<root>"


def _extract_validation_feedback(error: ValidationError) -> list[str]:
    """Extract concise validation feedback lines from a ``ValidationError``."""
    errors = error.errors()
    feedback_lines: list[str] = []
    for item in errors[:_MAX_VALIDATION_FEEDBACK_ITEMS]:
        loc_raw = item.get("loc", ())
        loc = tuple(loc_raw) if isinstance(loc_raw, (list, tuple)) else (loc_raw,)
        location = _format_validation_error_location(loc)
        message = str(item.get("msg", "Validation error"))
        feedback_lines.append(f"{location}: {message}")

    omitted = len(errors) - len(feedback_lines)
    if omitted > 0:
        feedback_lines.append(f"... and {omitted} more issues.")
    return feedback_lines


def _build_retry_instructions(base_instructions: str, error: ValidationError) -> str:
    """Append schema-validation feedback to instructions for a retry attempt."""
    feedback = _extract_validation_feedback(error)
    lines = [
        "--- PRIOR VALIDATION FEEDBACK ---",
        "The previous response failed schema validation.",
        "Fix ONLY the issues below and regenerate the full JSON response.",
    ]
    for i, issue in enumerate(feedback, start=1):
        lines.append(f"{i}. {issue}")
    lines.extend(
        [
            "Return exactly one assistant_messages item per input user message.",
            "Keep every assistant_messages.id aligned with the input id.",
            "Ensure each assistant_messages.body strictly matches the required schema and types.",
        ]
    )
    return base_instructions + "\n\n" + "\n".join(lines)


def _build_multimodal_retry_instructions(base_instructions: str, error: ValidationError) -> str:
    """Add validation feedback without the batched text response instructions."""
    lines = [
        "--- PRIOR VALIDATION FEEDBACK ---",
        "The previous response failed schema validation.",
        "Fix the issues below and regenerate a single response matching the required schema.",
    ]
    for i, issue in enumerate(_extract_validation_feedback(error), start=1):
        lines.append(f"{i}. {issue}")
    return base_instructions + "\n\n" + "\n".join(lines)


def _cache_key_for_input(value: str, *, multimodal: bool) -> str:
    """Version local files by content without conflating them with literal inputs."""
    if not multimodal:
        return value
    if "\0" in value:
        return _LITERAL_INPUT_PREFIX + value
    return local_file_cache_key(value)


def _resolve_cache_key(key: str) -> tuple[str, bytes | None]:
    """Return an input and its verified file snapshot on a cache miss."""
    if key.startswith(_LITERAL_INPUT_PREFIX):
        return key[len(_LITERAL_INPUT_PREFIX) :], None
    if "\0" in key:
        path, _ = key.rsplit("\0", 1)
        return path, read_local_file_for_cache_key(path, key)
    return key, None


def _ensure_files_unchanged(keys: list[str]) -> None:
    """Reject cache hits when a file changed after its initial key was computed."""
    for key in dict.fromkeys(keys):
        if key.startswith(_LITERAL_INPUT_PREFIX) or "\0" not in key:
            continue
        path, _ = key.rsplit("\0", 1)
        if local_file_cache_key(path) != key:
            raise ValueError(f"File changed during Responses request: {path}")


def _vectorize_system_message(system_message: str) -> str:
    """Build a system prompt that instructs the model to work on batched inputs.

    The returned XML‐ish prompt explains two things to the LLM:

    1. The *general* system instruction coming from the caller (`system_message`)
       is preserved verbatim.
    2. Extra instructions describe how the model should treat the incoming JSON
       that contains multiple user messages and how it must shape its output.

    Args:
        system_message (str): Single instance system instruction the caller would
            normally send to the model.

    Returns:
        str: Composite system prompt with embedded examples for the JSON‑mode
            endpoint (to be supplied via the ``instructions=`` field).
    """
    return f"""
<SystemMessage>
    <ElementInstructions>
        <Instruction>{system_message}</Instruction>
    </ElementInstructions>
    <BatchInstructions>
        <Instruction>
            You will receive multiple user messages at once.
            Please provide an appropriate response to each message individually.
        </Instruction>
    </BatchInstructions>
    <Examples>
        <Example>
            <Input>
                {{
                    "user_messages": [
                        {{
                            "id": 1,
                            "body": "{{user_message_1}}"
                        }},
                        {{
                            "id": 2,
                            "body": "{{user_message_2}}"
                        }}
                    ]
                }}
            </Input>
            <Output>
                {{
                    "assistant_messages": [
                        {{
                            "id": 1,
                            "body": "{{assistant_response_1}}"
                        }},
                        {{
                            "id": 2,
                            "body": "{{assistant_response_2}}"
                        }}
                    ]
                }}
            </Output>
        </Example>
    </Examples>
</SystemMessage>
"""


class Message(BaseModel, Generic[ResponseFormat]):
    id: int
    body: ResponseFormat


class Request(BaseModel):
    user_messages: list[Message[str]]


class Response(BaseModel, Generic[ResponseFormat]):
    assistant_messages: list[Message[ResponseFormat]]


@dataclass(frozen=True)
class BatchResponses(Generic[ResponseFormat]):
    """Stateless façade that turns OpenAI's JSON‑mode API into a batched API.

    This wrapper allows you to submit *multiple* user prompts in one JSON‑mode
    request and receive the answers in the original order.

    Example:
        ```python
        vector_llm = BatchResponses(
            client=openai_client,
            model_name="gpt-6-luna",
            system_message="You are a helpful assistant.",
            api_kwargs={"reasoning": {"effort": "none"}},
        )
        answers = vector_llm.parse(questions)
        ```

    Attributes:
        client (OpenAI): Initialised OpenAI client.
        model_name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name.
        system_message (str): System prompt prepended to every request.
        response_format (type[ResponseFormat]): Expected Pydantic model class or ``str`` for each assistant message.
        cache (BatchCache[str, ResponseFormat | None]): Order-preserving batching
            proxy with de-duplication and caching. Library-managed instances use
            bounded retention by default. Caller-provided caches must accept
            ``None`` because a response without parsed output is cached.
        max_validation_retries (int): Number of retries when structured output fails
            local schema validation.
        retry_policy (RetryPolicy | None): Transport limits. ``None`` preserves SDK retries.

    Notes:
        Internally the work is delegated to two helpers:

        * ``_predict_chunk`` – fragments the workload and restores ordering.
        * ``_request_llm`` – issues OpenAI API calls and retries with validation feedback when needed.
    """

    client: OpenAI
    model_name: str  # For Azure: deployment name, for OpenAI: model name
    system_message: str
    response_format: type[ResponseFormat] = str  # type: ignore[assignment]
    cache: BatchCache[str, ResponseFormat | None] = field(
        default_factory=lambda: BatchCache(batch_size=None, max_cache_size=DEFAULT_MANAGED_CACHE_SIZE)
    )
    api_kwargs: dict[str, Any] = field(default_factory=dict)
    max_validation_retries: int = 3
    multimodal: bool = False
    retry_policy: RetryPolicy | None = None
    limits: ResponseLimits = field(default_factory=ResponseLimits)
    _vectorized_system_message: str = field(init=False)
    _model_json_schema: dict = field(init=False)

    @classmethod
    def of(
        cls,
        client: OpenAI,
        model_name: str,
        system_message: str,
        response_format: type[ResponseFormat] = str,
        batch_size: int | None = None,
        max_validation_retries: int = 3,
        multimodal: bool = False,
        *,
        limits: ResponseLimits | None = None,
        retry_policy: RetryPolicy | None = None,
        **api_kwargs,
    ) -> "BatchResponses":
        """Factory constructor.

        Args:
            client (OpenAI): OpenAI client.
            model_name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name.
            system_message (str): System prompt for the model.
            response_format (type[ResponseFormat], optional): Expected output type. Defaults to ``str``.
            batch_size (int | None, optional): Max unique prompts per API call. Defaults to None
                (automatic batch size optimization). Set to a positive integer for fixed batch size.
            max_validation_retries (int, optional): Retry count when structured output fails local
                schema validation. Defaults to 3.
            multimodal (bool, optional): When ``True``, file paths and URLs in
                inputs are sent as multimodal content. Defaults to ``False``.
            limits (ResponseLimits | None): Text request context and item limits.
            retry_policy (RetryPolicy | None): Transport limits. ``None`` preserves SDK retries.
            **api_kwargs: Additional OpenAI API parameters (temperature, top_p, etc.).

        Returns:
            BatchResponses: Configured instance backed by a batching proxy.
        """
        return cls(
            client=client,
            model_name=model_name,
            system_message=system_message,
            response_format=response_format,
            cache=BatchCache(batch_size=batch_size, max_cache_size=DEFAULT_MANAGED_CACHE_SIZE),
            api_kwargs=api_kwargs,
            max_validation_retries=max_validation_retries,
            multimodal=multimodal,
            limits=limits if limits is not None else ResponseLimits(),
            retry_policy=retry_policy,
        )

    @classmethod
    def of_task(
        cls,
        client: OpenAI,
        model_name: str,
        task: PreparedTask[ResponseFormat],
        batch_size: int | None = None,
        max_validation_retries: int = 3,
        multimodal: bool = False,
        *,
        limits: ResponseLimits | None = None,
        retry_policy: RetryPolicy | None = None,
        **api_kwargs,
    ) -> "BatchResponses":
        """Factory from a PreparedTask.

        Args:
            client (OpenAI): OpenAI client.
            model_name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name.
            task (PreparedTask): Prepared task with instructions and response format.
            batch_size (int | None, optional): Max unique prompts per API call. Defaults to None
                (automatic batch size optimization). Set to a positive integer for fixed batch size.
            max_validation_retries (int, optional): Retry count when structured output fails local
                schema validation. Defaults to 3.
            multimodal (bool, optional): When ``True``, file paths and URLs in
                inputs are sent as multimodal content. Defaults to ``False``.
            limits (ResponseLimits | None): Text request context and item limits.
            retry_policy (RetryPolicy | None): Transport limits. ``None`` preserves SDK retries.
            **api_kwargs: Additional OpenAI API parameters forwarded to the Responses API.

        Returns:
            BatchResponses: Configured instance backed by a batching proxy.
        """
        return cls(
            client=client,
            model_name=model_name,
            system_message=task.instructions,
            response_format=task.response_format,
            cache=BatchCache(batch_size=batch_size, max_cache_size=DEFAULT_MANAGED_CACHE_SIZE),
            api_kwargs=api_kwargs,
            max_validation_retries=max_validation_retries,
            multimodal=multimodal,
            limits=limits if limits is not None else ResponseLimits(),
            retry_policy=retry_policy,
        )

    def __post_init__(self):
        if self.max_validation_retries < 0:
            raise ValueError("max_validation_retries must be >= 0")
        object.__setattr__(
            self,
            "_vectorized_system_message",
            _vectorize_system_message(self.system_message),
        )

    @observe(_LOGGER)
    def _request_llm(self, user_messages: list[Message[str]]) -> ParsedResponse[Response[ResponseFormat]]:
        """Call the OpenAI JSON‑mode endpoint, retrying on schema validation failures.

        Args:
            user_messages (list[Message[str]]): Sequence of ``Message[str]`` representing the
                prompts for this minibatch.  Each message carries a unique `id`
                so we can restore ordering later.

        Returns:
            ParsedResponse[Response[ResponseFormat]]: Parsed response containing assistant messages (arbitrary order).

        Raises:
            openai.RateLimitError: Re-raised after transport retries are exhausted.
            TimeoutError: The explicit transport deadline was exceeded.
            pydantic.ValidationError: Re‑raised when validation still fails after
                ``max_validation_retries`` correction attempts.
        """
        response_format = self.response_format

        class MessageT(BaseModel):
            id: int
            body: response_format  # type: ignore

        class ResponseT(BaseModel):
            assistant_messages: list[MessageT]

        instructions = self._vectorized_system_message
        input_json = Request(user_messages=user_messages).model_dump_json()
        deadline = retry_deadline(self.retry_policy)
        for attempt in range(self.max_validation_retries + 1):
            if attempt:
                _ensure_retry_fits_budget(
                    user_messages,
                    self.model_name,
                    instructions,
                    self.response_format,
                    self.limits,
                    self.api_kwargs,
                )
            try:
                response: ParsedResponse[ResponseT] = call_with_retry(
                    self.client,
                    self.retry_policy,
                    lambda client, options: client.responses.parse(
                        instructions=instructions,
                        model=self.model_name,
                        input=input_json,
                        text_format=ResponseT,
                        **options,
                    ),
                    self.api_kwargs,
                    deadline=deadline,
                )
                if response.output_parsed is not None:
                    _validate_response_ids(
                        [message.id for message in user_messages],
                        [message.id for message in response.output_parsed.assistant_messages],
                    )
                return cast(ParsedResponse[Response[ResponseFormat]], response)
            except ValidationError as e:
                if attempt >= self.max_validation_retries:
                    raise
                instructions = _build_retry_instructions(self._vectorized_system_message, e)

        raise RuntimeError("unreachable validation retry loop state")

    @observe(_LOGGER)
    def _request_multimodal(self, input_messages: ResponseInputParam) -> ResponseFormat | None:
        """Send a single multimodal request.

        Args:
            input_messages (ResponseInputParam): Pre-built input messages
                containing multimodal content (images, files, or audio).

        Returns:
            ResponseFormat | None: Parsed response or ``None``.
        """
        response_format: type[ResponseFormat] = self.response_format
        deadline = retry_deadline(self.retry_policy)

        if response_format is str:
            response: OAIResponse = call_with_retry(
                self.client,
                self.retry_policy,
                lambda client, options: client.responses.create(
                    instructions=self.system_message,
                    model=self.model_name,
                    input=input_messages,
                    **options,
                ),
                self.api_kwargs,
                deadline=deadline,
            )
            return cast(ResponseFormat, response.output_text)

        instructions = self.system_message
        for attempt in range(self.max_validation_retries + 1):
            try:
                parsed_response: ParsedResponse[ResponseFormat] = call_with_retry(
                    self.client,
                    self.retry_policy,
                    lambda client, options: client.responses.parse(
                        instructions=instructions,
                        model=self.model_name,
                        input=input_messages,
                        text_format=response_format,
                        **options,
                    ),
                    self.api_kwargs,
                    deadline=deadline,
                )
                return parsed_response.output_parsed
            except ValidationError as error:
                if attempt >= self.max_validation_retries:
                    raise
                instructions = _build_multimodal_retry_instructions(self.system_message, error)

        raise RuntimeError("unreachable validation retry loop state")

    @observe(_LOGGER)
    def _predict_chunk(self, user_messages: list[str]) -> list[ResponseFormat | None]:
        """Process a minibatch, routing inputs by type.

        When ``self.multimodal`` is ``False`` (default), all inputs are treated
        as plain text and batched via the JSON envelope.  When ``True``:

        * **Text-readable files** (source code, markup, etc.) are read as
          strings and batched together with plain text for dedup benefits.
        * **Binary files and images** are sent as individual multimodal
          requests via the Files API or inline base64.

        Args:
            user_messages (list[str]): Unique input strings for this minibatch.

        Returns:
            list[ResponseFormat | None]: Responses aligned to *user_messages*.
        """
        if not self.multimodal:
            messages: list[Message[str]] = [Message(id=i, body=m) for i, m in enumerate(user_messages)]
            return self._predict_text(messages)

        text_indices: list[int] = []
        multimodal_indices: list[int] = []
        resolved_inputs = [_resolve_cache_key(key) for key in user_messages]
        resolved_messages: list[str] = [msg for msg, _ in resolved_inputs]

        for i, (msg, file_bytes) in enumerate(resolved_inputs):
            if is_multimodal_input(msg):
                multimodal_indices.append(i)
            else:
                if is_readable_text_file(msg):
                    resolved_messages[i] = read_text_file(msg, file_bytes=file_bytes)
                text_indices.append(i)

        results: list[ResponseFormat | None] = [None] * len(user_messages)

        if text_indices:
            text_messages: list[Message[str]] = [
                Message(id=i, body=resolved_messages[idx]) for i, idx in enumerate(text_indices)
            ]
            text_results = self._predict_text(text_messages)
            for orig_idx, result in zip(text_indices, text_results):
                results[orig_idx] = result

        builder = MultimodalContentBuilder(client=self.client)
        for idx in multimodal_indices:
            msg, file_bytes = resolved_inputs[idx]
            input_messages, uploads = builder.build_with_uploads(msg, file_bytes=file_bytes)
            try:
                results[idx] = self._request_multimodal(input_messages)
            finally:
                builder.cleanup_uploads(uploads)

        return results

    def _predict_text(self, messages: list[Message[str]]) -> list[ResponseFormat | None]:
        batches = _plan_response_batches(
            messages,
            self.model_name,
            self._vectorized_system_message,
            self.response_format,
            self.limits,
            self.max_validation_retries,
            self.api_kwargs,
        )
        results: dict[int, ResponseFormat] = {}
        for batch in batches:
            response = self._request_llm(batch)
            if response.output_parsed:
                results.update({message.id: message.body for message in response.output_parsed.assistant_messages})
        return [results.get(message.id) for message in messages]

    @observe(_LOGGER)
    def parse(self, inputs: list[str]) -> list[ResponseFormat | None]:
        """Batched predict.

        Args:
            inputs (list[str]): Prompts that require responses. Duplicates are de‑duplicated.

        Returns:
            list[ResponseFormat | None]: Assistant responses aligned to ``inputs``.
                An absent parsed result is returned as ``None`` and cached.
        """
        keys = [_cache_key_for_input(value, multimodal=self.multimodal) for value in inputs]
        results = self.cache.map(keys, self._predict_chunk)
        if self.multimodal:
            _ensure_files_unchanged(keys)
        return results


@dataclass(frozen=True)
class AsyncBatchResponses(Generic[ResponseFormat]):
    """Stateless façade that turns OpenAI's JSON-mode API into a batched API (Async version).

    This wrapper allows you to submit *multiple* user prompts in one JSON-mode
    request and receive the answers in the original order asynchronously. It also
    controls the maximum number of concurrent requests to the OpenAI API.

    Example:
        ```python
        import asyncio
        from openai import AsyncOpenAI
        from openaivec import AsyncBatchResponses

        openai_async_client = AsyncOpenAI()  # initialize your client

        vector_llm = AsyncBatchResponses.of(
            client=openai_async_client,
            model_name="gpt-6-luna",
            system_message="You are a helpful assistant.",
            batch_size=64,
            max_concurrency=5,
            reasoning={"effort": "none"},
        )
        questions = [
            "What is the capital of France?",
            "Explain quantum physics simply.",
        ]

        async def main():
            answers = await vector_llm.parse(questions)
            print(answers)

        asyncio.run(main())
        ```

    Attributes:
        client (AsyncOpenAI): Initialised OpenAI async client.
        model_name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name.
        system_message (str): System prompt prepended to every request.
        response_format (type[ResponseFormat]): Expected Pydantic model class or ``str`` for each assistant message.
        cache (AsyncBatchCache[str, ResponseFormat | None]): Async batching proxy
            with de-duplication and concurrency control. Library-managed
            instances use bounded retention by default. Caller-provided caches
            must accept ``None`` because a response without parsed output is cached.
        max_validation_retries (int): Number of retries when structured output fails
            local schema validation.
        retry_policy (RetryPolicy | None): Transport limits. ``None`` preserves SDK retries.
    """

    client: AsyncOpenAI
    model_name: str  # For Azure: deployment name, for OpenAI: model name
    system_message: str
    response_format: type[ResponseFormat] = str  # type: ignore[assignment]
    cache: AsyncBatchCache[str, ResponseFormat | None] = field(
        default_factory=lambda: AsyncBatchCache(
            batch_size=None,
            max_concurrency=8,
            max_cache_size=DEFAULT_MANAGED_CACHE_SIZE,
        )
    )
    api_kwargs: dict[str, Any] = field(default_factory=dict)
    max_validation_retries: int = 3
    multimodal: bool = False
    retry_policy: RetryPolicy | None = None
    limits: ResponseLimits = field(default_factory=ResponseLimits)
    _vectorized_system_message: str = field(init=False)
    _model_json_schema: dict = field(init=False)
    _media_semaphore: asyncio.Semaphore = field(init=False, repr=False, compare=False)

    @classmethod
    def of(
        cls,
        client: AsyncOpenAI,
        model_name: str,
        system_message: str,
        response_format: type[ResponseFormat] = str,
        batch_size: int | None = None,
        max_concurrency: int = 8,
        max_validation_retries: int = 3,
        multimodal: bool = False,
        *,
        limits: ResponseLimits | None = None,
        retry_policy: RetryPolicy | None = None,
        **api_kwargs,
    ) -> "AsyncBatchResponses":
        """Factory constructor.

        Args:
            client (AsyncOpenAI): OpenAI async client.
            model_name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name.
            system_message (str): System prompt.
            response_format (type[ResponseFormat], optional): Expected output type. Defaults to ``str``.
            batch_size (int | None, optional): Max unique prompts per API call. Defaults to None
                (automatic batch size optimization). Set to a positive integer for fixed batch size.
            max_concurrency (int, optional): Max concurrent API calls. Defaults to 8.
            max_validation_retries (int, optional): Retry count when structured output fails local
                schema validation. Defaults to 3.
            multimodal (bool, optional): When ``True``, file paths and URLs in
                inputs are sent as multimodal content. Defaults to ``False``.
            limits (ResponseLimits | None): Text request context and item limits.
            retry_policy (RetryPolicy | None): Transport limits. ``None`` preserves SDK retries.
            **api_kwargs: Additional OpenAI API parameters (temperature, top_p, etc.).

        Returns:
            AsyncBatchResponses: Configured instance backed by an async batching proxy.
        """
        return cls(
            client=client,
            model_name=model_name,
            system_message=system_message,
            response_format=response_format,
            cache=AsyncBatchCache(
                batch_size=batch_size,
                max_concurrency=max_concurrency,
                max_cache_size=DEFAULT_MANAGED_CACHE_SIZE,
            ),
            api_kwargs=api_kwargs,
            max_validation_retries=max_validation_retries,
            multimodal=multimodal,
            limits=limits if limits is not None else ResponseLimits(),
            retry_policy=retry_policy,
        )

    @classmethod
    def of_task(
        cls,
        client: AsyncOpenAI,
        model_name: str,
        task: PreparedTask[ResponseFormat],
        batch_size: int | None = None,
        max_concurrency: int = 8,
        max_validation_retries: int = 3,
        multimodal: bool = False,
        *,
        limits: ResponseLimits | None = None,
        retry_policy: RetryPolicy | None = None,
        **api_kwargs,
    ) -> "AsyncBatchResponses":
        """Factory from a PreparedTask.

        Args:
            client (AsyncOpenAI): OpenAI async client.
            model_name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name.
            task (PreparedTask): Prepared task with instructions and response format.
            batch_size (int | None, optional): Max unique prompts per API call. Defaults to None
                (automatic batch size optimization). Set to a positive integer for fixed batch size.
            max_concurrency (int, optional): Max concurrent API calls. Defaults to 8.
            max_validation_retries (int, optional): Retry count when structured output fails local
                schema validation. Defaults to 3.
            multimodal (bool, optional): When ``True``, file paths and URLs in
                inputs are sent as multimodal content. Defaults to ``False``.
            limits (ResponseLimits | None): Text request context and item limits.
            retry_policy (RetryPolicy | None): Transport limits. ``None`` preserves SDK retries.
            **api_kwargs: Additional OpenAI API parameters forwarded to the Responses API.

        Returns:
            AsyncBatchResponses: Configured instance backed by an async batching proxy.
        """
        return cls(
            client=client,
            model_name=model_name,
            system_message=task.instructions,
            response_format=task.response_format,
            cache=AsyncBatchCache(
                batch_size=batch_size,
                max_concurrency=max_concurrency,
                max_cache_size=DEFAULT_MANAGED_CACHE_SIZE,
            ),
            api_kwargs=api_kwargs,
            max_validation_retries=max_validation_retries,
            multimodal=multimodal,
            limits=limits if limits is not None else ResponseLimits(),
            retry_policy=retry_policy,
        )

    def __post_init__(self):
        if self.max_validation_retries < 0:
            raise ValueError("max_validation_retries must be >= 0")
        object.__setattr__(
            self,
            "_vectorized_system_message",
            _vectorize_system_message(self.system_message),
        )
        object.__setattr__(self, "_media_semaphore", asyncio.Semaphore(self.cache.max_concurrency))

    @observe(_LOGGER)
    async def _request_llm(self, user_messages: list[Message[str]]) -> ParsedResponse[Response[ResponseFormat]]:
        """Call the OpenAI JSON‑mode endpoint asynchronously with validation retries.

        Args:
            user_messages (list[Message[str]]): Sequence of ``Message[str]`` representing the minibatch prompts.

        Returns:
            ParsedResponse[Response[ResponseFormat]]: Parsed response with assistant messages (arbitrary order).

        Raises:
            openai.RateLimitError: Re-raised after transport retries are exhausted.
            TimeoutError: The explicit transport deadline was exceeded.
            pydantic.ValidationError: Re‑raised when validation still fails after
                ``max_validation_retries`` correction attempts.
        """
        response_format = self.response_format

        class MessageT(BaseModel):
            id: int
            body: response_format  # type: ignore

        class ResponseT(BaseModel):
            assistant_messages: list[MessageT]

        instructions = self._vectorized_system_message
        input_json = Request(user_messages=user_messages).model_dump_json()
        deadline = retry_deadline(self.retry_policy)
        for attempt in range(self.max_validation_retries + 1):
            if attempt:
                _ensure_retry_fits_budget(
                    user_messages,
                    self.model_name,
                    instructions,
                    self.response_format,
                    self.limits,
                    self.api_kwargs,
                )
            try:
                response: ParsedResponse[ResponseT] = await call_with_retry_async(
                    self.client,
                    self.retry_policy,
                    lambda client, options: client.responses.parse(
                        instructions=instructions,
                        model=self.model_name,
                        input=input_json,
                        text_format=ResponseT,
                        **options,
                    ),
                    self.api_kwargs,
                    deadline=deadline,
                )
                if response.output_parsed is not None:
                    _validate_response_ids(
                        [message.id for message in user_messages],
                        [message.id for message in response.output_parsed.assistant_messages],
                    )
                return cast(ParsedResponse[Response[ResponseFormat]], response)
            except ValidationError as e:
                if attempt >= self.max_validation_retries:
                    raise
                instructions = _build_retry_instructions(self._vectorized_system_message, e)

        raise RuntimeError("unreachable validation retry loop state")

    @observe(_LOGGER)
    async def _request_multimodal(self, input_messages: ResponseInputParam) -> ResponseFormat | None:
        """Send a single multimodal request (async).

        Args:
            input_messages (ResponseInputParam): Pre-built input messages
                containing multimodal content (images, files, or audio).

        Returns:
            ResponseFormat | None: Parsed response or ``None``.
        """
        response_format: type[ResponseFormat] = self.response_format
        deadline = retry_deadline(self.retry_policy)

        if response_format is str:
            response: OAIResponse = await call_with_retry_async(
                self.client,
                self.retry_policy,
                lambda client, options: client.responses.create(
                    instructions=self.system_message,
                    model=self.model_name,
                    input=input_messages,
                    **options,
                ),
                self.api_kwargs,
                deadline=deadline,
            )
            return cast(ResponseFormat, response.output_text)

        instructions = self.system_message
        for attempt in range(self.max_validation_retries + 1):
            try:
                parsed_response: ParsedResponse[ResponseFormat] = await call_with_retry_async(
                    self.client,
                    self.retry_policy,
                    lambda client, options: client.responses.parse(
                        instructions=instructions,
                        model=self.model_name,
                        input=input_messages,
                        text_format=response_format,
                        **options,
                    ),
                    self.api_kwargs,
                    deadline=deadline,
                )
                return parsed_response.output_parsed
            except ValidationError as error:
                if attempt >= self.max_validation_retries:
                    raise
                instructions = _build_multimodal_retry_instructions(self.system_message, error)

        raise RuntimeError("unreachable validation retry loop state")

    async def _predict_media(self, value: str, file_bytes: bytes | None) -> ResponseFormat | None:
        async with self._media_semaphore:
            builder = AsyncMultimodalContentBuilder(client=self.client)
            input_messages, uploads = await builder.build_with_uploads(value, file_bytes=file_bytes)
            try:
                return await self._request_multimodal(input_messages)
            finally:
                await builder.cleanup_uploads(uploads)

    @observe(_LOGGER)
    async def _predict_chunk(self, user_messages: list[str]) -> list[ResponseFormat | None]:
        """Process a minibatch, routing inputs by type (async).

        When ``self.multimodal`` is ``False`` (default), all inputs are treated
        as plain text.  When ``True``, text-readable files are inlined and
        batched; binary files and images use individual multimodal calls.

        Args:
            user_messages (list[str]): Unique input strings for this minibatch.

        Returns:
            list[ResponseFormat | None]: Responses aligned to *user_messages*.
        """
        if not self.multimodal:
            messages: list[Message[str]] = [Message(id=i, body=m) for i, m in enumerate(user_messages)]
            return await self._predict_text(messages)

        text_indices: list[int] = []
        multimodal_indices: list[int] = []
        resolved_inputs = [_resolve_cache_key(key) for key in user_messages]
        resolved_messages: list[str] = [msg for msg, _ in resolved_inputs]

        for i, (msg, file_bytes) in enumerate(resolved_inputs):
            if is_multimodal_input(msg):
                multimodal_indices.append(i)
            else:
                if is_readable_text_file(msg):
                    resolved_messages[i] = read_text_file(msg, file_bytes=file_bytes)
                text_indices.append(i)

        results: list[ResponseFormat | None] = [None] * len(user_messages)

        if text_indices:
            text_messages: list[Message[str]] = [
                Message(id=i, body=resolved_messages[idx]) for i, idx in enumerate(text_indices)
            ]
            text_results = await self._predict_text(text_messages)
            for orig_idx, result in zip(text_indices, text_results):
                results[orig_idx] = result

        if multimodal_indices:
            tasks = [asyncio.create_task(self._predict_media(*resolved_inputs[idx])) for idx in multimodal_indices]
            cancelled = False
            try:
                remaining = set(tasks)
                while remaining:
                    finished, remaining = await asyncio.wait(remaining, return_when=asyncio.FIRST_COMPLETED)
                    if any(task.cancelled() or task.exception() is not None for task in finished):
                        break
            except asyncio.CancelledError:
                cancelled = True
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                pending = asyncio.gather(*tasks, return_exceptions=True)
                while True:
                    try:
                        media_results = await asyncio.shield(pending)
                        break
                    except asyncio.CancelledError:
                        cancelled = True
            errors = [outcome for outcome in media_results if isinstance(outcome, Exception)]
            if len(errors) > 1:
                raise RuntimeError(
                    f"Multimodal requests failed: {'; '.join(str(error) for error in errors)}"
                ) from errors[0]
            if errors:
                raise errors[0]
            if cancelled or any(isinstance(outcome, asyncio.CancelledError) for outcome in media_results):
                raise asyncio.CancelledError
            for idx, result in zip(multimodal_indices, media_results):
                results[idx] = cast(ResponseFormat | None, result)

        return results

    async def _predict_text(self, messages: list[Message[str]]) -> list[ResponseFormat | None]:
        batches = _plan_response_batches(
            messages,
            self.model_name,
            self._vectorized_system_message,
            self.response_format,
            self.limits,
            self.max_validation_retries,
            self.api_kwargs,
        )
        results: dict[int, ResponseFormat] = {}
        for batch in batches:
            response = await self._request_llm(batch)
            if response.output_parsed:
                results.update({message.id: message.body for message in response.output_parsed.assistant_messages})
        return [results.get(message.id) for message in messages]

    @observe(_LOGGER)
    async def parse(self, inputs: list[str]) -> list[ResponseFormat | None]:
        """Batched predict (async).

        Args:
            inputs (list[str]): Prompts that require responses. Duplicates are de‑duplicated.

        Returns:
            list[ResponseFormat | None]: Assistant responses aligned to ``inputs``.
                An absent parsed result is returned as ``None`` and cached.
        """
        keys = [_cache_key_for_input(value, multimodal=self.multimodal) for value in inputs]
        results = await self.cache.map(keys, self._predict_chunk)
        if self.multimodal:
            _ensure_files_unchanged(keys)
        return results
