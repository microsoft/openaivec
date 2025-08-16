import warnings
from dataclasses import dataclass, field
from logging import Logger, getLogger
from typing import Any, Generic, List, Type, cast

from openai import AsyncOpenAI, BadRequestError, InternalServerError, OpenAI, RateLimitError
from openai.types.responses import ParsedResponse
from pydantic import BaseModel

from openaivec._log import observe
from openaivec._model import PreparedTask, ResponseFormat
from openaivec._proxy import AsyncBatchingMapProxy, BatchingMapProxy
from openaivec._util import backoff, backoff_async

__all__ = [
    "BatchResponses",
    "AsyncBatchResponses",
]

_LOGGER: Logger = getLogger(__name__)


def _handle_temperature_error(error: BadRequestError, model_name: str, temperature: float) -> None:
    """Handle temperature-related errors for reasoning models.

    Detects when a model doesn't support temperature parameter and provides guidance.

    Args:
        error (BadRequestError): The OpenAI API error.
        model_name (str): The model that caused the error.
        temperature (float): The temperature value that was rejected.
    """
    error_message = str(error)
    if "temperature" in error_message.lower() and "not supported" in error_message.lower():
        guidance_message = (
            f"🔧 Model '{model_name}' rejected temperature parameter (value: {temperature}). "
            f"This typically happens with reasoning models (o1-preview, o1-mini, o3, etc.). "
            f"To fix this, you MUST explicitly set temperature=None:\n"
            f"• For pandas: df.col.ai.responses('prompt', temperature=None)\n"
            f"• For Spark UDFs: responses_udf('prompt', temperature=None)\n"
            f"• For direct API: BatchResponses.of(client, model, temperature=None)\n"
            f"• Original error: {error_message}\n"
            f"See: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reasoning"
        )
        warnings.warn(guidance_message, UserWarning, stacklevel=5)

        # Re-raise with enhanced message
        enhanced_message = f"{error_message}\n\nSUGGESTION: Set temperature=None to resolve this error."
        raise BadRequestError(message=enhanced_message, response=error.response, body=error.body)


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
    user_messages: List[Message[str]]


class Response(BaseModel, Generic[ResponseFormat]):
    assistant_messages: List[Message[ResponseFormat]]


@dataclass(frozen=True)
class BatchResponses(Generic[ResponseFormat]):
    """Stateless façade that turns OpenAI's JSON‑mode API into a batched API.

    This wrapper allows you to submit *multiple* user prompts in one JSON‑mode
    request and receive the answers in the original order.

    Example:
        ```python
        vector_llm = BatchResponses(
            client=openai_client,
            model_name="gpt‑4o‑mini",
            system_message="You are a helpful assistant."
        )
        answers = vector_llm.parse(questions)
        ```

    Attributes:
        client (OpenAI): Initialised OpenAI client.
        model_name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name.
        system_message (str): System prompt prepended to every request.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus‑sampling parameter.
        response_format (Type[ResponseFormat]): Expected Pydantic model class or ``str`` for each assistant message.
        cache (BatchingMapProxy[str, ResponseFormat]): Order‑preserving batching proxy with de‑duplication and caching.

    Notes:
        Internally the work is delegated to two helpers:

        * ``_predict_chunk`` – fragments the workload and restores ordering.
        * ``_request_llm`` – performs a single OpenAI API call.
    """

    client: OpenAI
    model_name: str  # For Azure: deployment name, for OpenAI: model name
    system_message: str
    temperature: float | None = None
    top_p: float = 1.0
    response_format: Type[ResponseFormat] = str  # type: ignore[assignment]
    cache: BatchingMapProxy[str, ResponseFormat] = field(default_factory=lambda: BatchingMapProxy(batch_size=None))
    _vectorized_system_message: str = field(init=False)
    _model_json_schema: dict = field(init=False)

    @classmethod
    def of(
        cls,
        client: OpenAI,
        model_name: str,
        system_message: str,
        temperature: float | None = 0.0,
        top_p: float = 1.0,
        response_format: Type[ResponseFormat] = str,
        batch_size: int | None = None,
    ) -> "BatchResponses":
        """Factory constructor.

        Args:
            client (OpenAI): OpenAI client.
            model_name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name.
            system_message (str): System prompt for the model.
            temperature (float, optional): Sampling temperature. Defaults to 0.0.
            top_p (float, optional): Nucleus sampling parameter. Defaults to 1.0.
            response_format (Type[ResponseFormat], optional): Expected output type. Defaults to ``str``.
            batch_size (int | None, optional): Max unique prompts per API call. Defaults to None
                (automatic batch size optimization). Set to a positive integer for fixed batch size.

        Returns:
            BatchResponses: Configured instance backed by a batching proxy.
        """
        return cls(
            client=client,
            model_name=model_name,
            system_message=system_message,
            temperature=temperature,
            top_p=top_p,
            response_format=response_format,
            cache=BatchingMapProxy(batch_size=batch_size),
        )

    @classmethod
    def of_task(
        cls, client: OpenAI, model_name: str, task: PreparedTask[ResponseFormat], batch_size: int | None = None
    ) -> "BatchResponses":
        """Factory from a PreparedTask.

        Args:
            client (OpenAI): OpenAI client.
            model_name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name.
            task (PreparedTask): Prepared task with instructions and response format.
            batch_size (int | None, optional): Max unique prompts per API call. Defaults to None
                (automatic batch size optimization). Set to a positive integer for fixed batch size.

        Returns:
            BatchResponses: Configured instance backed by a batching proxy.
        """
        return cls(
            client=client,
            model_name=model_name,
            system_message=task.instructions,
            temperature=task.temperature,
            top_p=task.top_p,
            response_format=task.response_format,
            cache=BatchingMapProxy(batch_size=batch_size),
        )

    def __post_init__(self):
        object.__setattr__(
            self,
            "_vectorized_system_message",
            _vectorize_system_message(self.system_message),
        )

    @observe(_LOGGER)
    @backoff(exceptions=[RateLimitError, InternalServerError], scale=1, max_retries=12)
    def _request_llm(
        self, user_messages: List[Message[str]], **extra_api_params: Any
    ) -> ParsedResponse[Response[ResponseFormat]]:
        """Make a single call to the OpenAI JSON‑mode endpoint.

        Args:
            user_messages (List[Message[str]]): Sequence of ``Message[str]`` representing the
                prompts for this minibatch.  Each message carries a unique `id`
                so we can restore ordering later.

        Returns:
            ParsedResponse[Response[ResponseFormat]]: Parsed response containing assistant messages (arbitrary order).

        Raises:
            openai.RateLimitError: Transparently re‑raised after the
                exponential back‑off decorator exhausts all retries.
        """
        response_format = self.response_format

        class MessageT(BaseModel):
            id: int
            body: response_format  # type: ignore

        class ResponseT(BaseModel):
            assistant_messages: List[MessageT]

        # Build base API parameters (cannot be overridden by caller)
        api_params: dict[str, Any] = {
            "model": self.model_name,
            "instructions": self._vectorized_system_message,
            "input": Request(user_messages=user_messages).model_dump_json(),
            "text_format": ResponseT,
        }

        # Resolve nucleus sampling (caller can override)
        top_p = extra_api_params.pop("top_p", self.top_p)
        if top_p is not None:
            api_params["top_p"] = top_p

        # Resolve temperature (caller can override). If None, omit entirely for reasoning models.
        temperature = extra_api_params.pop("temperature", self.temperature)
        if temperature is not None:
            api_params["temperature"] = temperature

        # Merge remaining user supplied params, excluding protected keys
        for k, v in extra_api_params.items():
            if k in {"model", "instructions", "input", "text_format"}:
                continue  # ignore attempts to override core batching contract
            api_params[k] = v

        try:
            completion: ParsedResponse[ResponseT] = self.client.responses.parse(**api_params)
        except BadRequestError as e:
            _handle_temperature_error(e, self.model_name, self.temperature or 0.0)
            raise  # Re-raise if it wasn't a temperature error

        return cast(ParsedResponse[Response[ResponseFormat]], completion)

    @observe(_LOGGER)
    def _predict_chunk(self, user_messages: List[str], **api_kwargs: Any) -> List[ResponseFormat | None]:
        """Helper executed for every unique minibatch.

            This method:
            1. Converts plain strings into `Message[str]` with stable indices.
            2. Delegates the request to `_request_llm`.
            3. Reorders the responses so they match the original indices.

        The function is pure – it has no side‑effects and the result depends
        only on its arguments – which allows safe reuse.
        """
        messages = [Message(id=i, body=message) for i, message in enumerate(user_messages)]
        responses: ParsedResponse[Response[ResponseFormat]] = self._request_llm(messages, **api_kwargs)
        if not responses.output_parsed:
            return [None] * len(messages)
        response_dict = {message.id: message.body for message in responses.output_parsed.assistant_messages}
        sorted_responses: List[ResponseFormat | None] = [response_dict.get(m.id, None) for m in messages]
        return sorted_responses

    @observe(_LOGGER)
    def parse(self, inputs: List[str], **api_kwargs: Any) -> List[ResponseFormat | None]:
        """Batched predict.

        Accepts arbitrary keyword arguments that are forwarded to the underlying
        ``OpenAI.responses.parse`` call for future‑proofing (e.g., ``max_output_tokens``,
        penalties, etc.). ``top_p`` and ``temperature`` default to the instance's
        configured values but can be overridden explicitly.

        Args:
            inputs (List[str]): Prompts that require responses. Duplicates are de‑duplicated.
            **api_kwargs: Extra keyword args forwarded to the OpenAI Responses API.

        Returns:
            List[ResponseFormat | None]: Assistant responses aligned to ``inputs``.
        """
        if not api_kwargs:
            return self.cache.map(inputs, self._predict_chunk)  # type: ignore[return-value]

        def _predict_with(xs: List[str]) -> List[ResponseFormat | None]:
            return self._predict_chunk(xs, **api_kwargs)

        return self.cache.map(inputs, _predict_with)  # type: ignore[return-value]


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
            model_name="gpt-4.1-mini",
            system_message="You are a helpful assistant.",
            batch_size=64,
            max_concurrency=5,
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
        temperature (float): Sampling temperature.
        top_p (float): Nucleus‑sampling parameter.
        response_format (Type[ResponseFormat]): Expected Pydantic model class or ``str`` for each assistant message.
        cache (AsyncBatchingMapProxy[str, ResponseFormat]): Async batching proxy with de‑duplication
            and concurrency control.
    """

    client: AsyncOpenAI
    model_name: str  # For Azure: deployment name, for OpenAI: model name
    system_message: str
    temperature: float | None = 0.0
    top_p: float = 1.0
    response_format: Type[ResponseFormat] = str  # type: ignore[assignment]
    cache: AsyncBatchingMapProxy[str, ResponseFormat] = field(
        default_factory=lambda: AsyncBatchingMapProxy(batch_size=None, max_concurrency=8)
    )
    _vectorized_system_message: str = field(init=False)
    _model_json_schema: dict = field(init=False)

    @classmethod
    def of(
        cls,
        client: AsyncOpenAI,
        model_name: str,
        system_message: str,
        temperature: float | None = None,
        top_p: float = 1.0,
        response_format: Type[ResponseFormat] = str,
        batch_size: int | None = None,
        max_concurrency: int = 8,
    ) -> "AsyncBatchResponses":
        """Factory constructor.

        Args:
            client (AsyncOpenAI): OpenAI async client.
            model_name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name.
            system_message (str): System prompt.
            temperature (float, optional): Sampling temperature. Defaults to 0.0.
            top_p (float, optional): Nucleus sampling parameter. Defaults to 1.0.
            response_format (Type[ResponseFormat], optional): Expected output type. Defaults to ``str``.
            batch_size (int | None, optional): Max unique prompts per API call. Defaults to None
                (automatic batch size optimization). Set to a positive integer for fixed batch size.
            max_concurrency (int, optional): Max concurrent API calls. Defaults to 8.

        Returns:
            AsyncBatchResponses: Configured instance backed by an async batching proxy.
        """
        return cls(
            client=client,
            model_name=model_name,
            system_message=system_message,
            temperature=temperature,
            top_p=top_p,
            response_format=response_format,
            cache=AsyncBatchingMapProxy(batch_size=batch_size, max_concurrency=max_concurrency),
        )

    @classmethod
    def of_task(
        cls,
        client: AsyncOpenAI,
        model_name: str,
        task: PreparedTask[ResponseFormat],
        batch_size: int | None = None,
        max_concurrency: int = 8,
    ) -> "AsyncBatchResponses":
        """Factory from a PreparedTask.

        Args:
            client (AsyncOpenAI): OpenAI async client.
            model_name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name.
            task (PreparedTask): Prepared task with instructions and response format.
            batch_size (int | None, optional): Max unique prompts per API call. Defaults to None
                (automatic batch size optimization). Set to a positive integer for fixed batch size.
            max_concurrency (int, optional): Max concurrent API calls. Defaults to 8.

        Returns:
            AsyncBatchResponses: Configured instance backed by an async batching proxy.
        """
        return cls(
            client=client,
            model_name=model_name,
            system_message=task.instructions,
            temperature=task.temperature,
            top_p=task.top_p,
            response_format=task.response_format,
            cache=AsyncBatchingMapProxy(batch_size=batch_size, max_concurrency=max_concurrency),
        )

    def __post_init__(self):
        object.__setattr__(
            self,
            "_vectorized_system_message",
            _vectorize_system_message(self.system_message),
        )

    @backoff_async(exceptions=[RateLimitError, InternalServerError], scale=1, max_retries=12)
    @observe(_LOGGER)
    async def _request_llm(
        self, user_messages: List[Message[str]], **extra_api_params: Any
    ) -> ParsedResponse[Response[ResponseFormat]]:
        """Make a single async call to the OpenAI JSON‑mode endpoint.

        Args:
            user_messages (List[Message[str]]): Sequence of ``Message[str]`` representing the minibatch prompts.

        Returns:
            ParsedResponse[Response[ResponseFormat]]: Parsed response with assistant messages (arbitrary order).

        Raises:
            RateLimitError: Re‑raised after back‑off retries are exhausted.
        """
        response_format = self.response_format

        class MessageT(BaseModel):
            id: int
            body: response_format  # type: ignore

        class ResponseT(BaseModel):
            assistant_messages: List[MessageT]

        # Build base API parameters (cannot be overridden by caller)
        api_params: dict[str, Any] = {
            "model": self.model_name,
            "instructions": self._vectorized_system_message,
            "input": Request(user_messages=user_messages).model_dump_json(),
            "text_format": ResponseT,
        }

        # Resolve nucleus sampling (caller can override)
        top_p = extra_api_params.pop("top_p", self.top_p)
        if top_p is not None:
            api_params["top_p"] = top_p

        # Resolve temperature (caller can override). If None, omit entirely for reasoning models.
        temperature = extra_api_params.pop("temperature", self.temperature)
        if temperature is not None:
            api_params["temperature"] = temperature

        # Merge remaining user supplied params, excluding protected keys
        for k, v in extra_api_params.items():
            if k in {"model", "instructions", "input", "text_format"}:
                continue
            api_params[k] = v

        try:
            completion: ParsedResponse[ResponseT] = await self.client.responses.parse(**api_params)
        except BadRequestError as e:
            _handle_temperature_error(e, self.model_name, self.temperature or 0.0)
            raise  # Re-raise if it wasn't a temperature error

        return cast(ParsedResponse[Response[ResponseFormat]], completion)

    @observe(_LOGGER)
    async def _predict_chunk(self, user_messages: List[str], **api_kwargs: Any) -> List[ResponseFormat | None]:
        """Async helper executed for every unique minibatch.

            This method:
            1. Converts plain strings into `Message[str]` with stable indices.
            2. Delegates the request to `_request_llm`.
            3. Reorders the responses so they match the original indices.

        The function is pure – it has no side‑effects and the result depends only on its arguments.
        """
        messages = [Message(id=i, body=message) for i, message in enumerate(user_messages)]
        responses: ParsedResponse[Response[ResponseFormat]] = await self._request_llm(messages, **api_kwargs)  # type: ignore[call-issue]
        if not responses.output_parsed:
            return [None] * len(messages)
        response_dict = {message.id: message.body for message in responses.output_parsed.assistant_messages}
        # Ensure proper handling for missing IDs - this shouldn't happen in normal operation
        sorted_responses: List[ResponseFormat | None] = [response_dict.get(m.id, None) for m in messages]
        return sorted_responses

    @observe(_LOGGER)
    async def parse(self, inputs: List[str], **api_kwargs: Any) -> List[ResponseFormat | None]:
        """Batched predict (async).

        Accepts arbitrary keyword arguments forwarded to ``AsyncOpenAI.responses.parse``.
        ``top_p`` and ``temperature`` default to instance configuration but can be
        overridden per call. This prepares for future API parameters without
        changing the public surface again.

        Args:
            inputs (List[str]): Prompts that require responses. Duplicates are de‑duplicated.
            **api_kwargs: Extra keyword args for the OpenAI Responses API.

        Returns:
            List[ResponseFormat | None]: Assistant responses aligned to ``inputs``.
        """
        if not api_kwargs:
            return await self.cache.map(inputs, self._predict_chunk)  # type: ignore[return-value]

        async def _predict_with(xs: List[str]) -> List[ResponseFormat | None]:
            return await self._predict_chunk(xs, **api_kwargs)

        return await self.cache.map(inputs, _predict_with)  # type: ignore[return-value]
