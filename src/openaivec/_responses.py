import warnings
from dataclasses import dataclass, field
from logging import Logger, getLogger
from typing import Generic, cast

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
            model_name="gpt‑4o‑mini",
            system_message="You are a helpful assistant."
        )
        answers = vector_llm.parse(questions)
        ```

    Attributes:
        client (OpenAI): Initialised OpenAI client.
        model_name (str): For Azure OpenAI, use your deployment name. For OpenAI, use the model name.
        system_message (str): System prompt prepended to every request.
        response_format (type[ResponseFormat]): Expected Pydantic model class or ``str`` for each assistant message.
        cache (BatchingMapProxy[str, ResponseFormat]): Order‑preserving batching proxy with de‑duplication and caching.

    Notes:
        Internally the work is delegated to two helpers:

        * ``_predict_chunk`` – fragments the workload and restores ordering.
        * ``_request_llm`` – performs a single OpenAI API call.
    """

    client: OpenAI
    model_name: str  # For Azure: deployment name, for OpenAI: model name
    system_message: str
    response_format: type[ResponseFormat] = str  # type: ignore[assignment]
    cache: BatchingMapProxy[str, ResponseFormat] = field(default_factory=lambda: BatchingMapProxy(batch_size=None))
    api_kwargs: dict[str, int | float | str | bool] = field(default_factory=dict)
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
            **api_kwargs: Additional OpenAI API parameters (temperature, top_p, etc.).

        Returns:
            BatchResponses: Configured instance backed by a batching proxy.
        """
        return cls(
            client=client,
            model_name=model_name,
            system_message=system_message,
            response_format=response_format,
            cache=BatchingMapProxy(batch_size=batch_size),
            api_kwargs=api_kwargs,
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
            response_format=task.response_format,
            cache=BatchingMapProxy(batch_size=batch_size),
            api_kwargs=task.api_kwargs,
        )

    def __post_init__(self):
        object.__setattr__(
            self,
            "_vectorized_system_message",
            _vectorize_system_message(self.system_message),
        )

    @observe(_LOGGER)
    @backoff(exceptions=[RateLimitError, InternalServerError], scale=1, max_retries=12)
    def _request_llm(self, user_messages: list[Message[str]]) -> ParsedResponse[Response[ResponseFormat]]:
        """Make a single call to the OpenAI JSON‑mode endpoint.

        Args:
            user_messages (list[Message[str]]): Sequence of ``Message[str]`` representing the
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
            assistant_messages: list[MessageT]

        try:
            response: ParsedResponse[ResponseT] = self.client.responses.parse(
                instructions=self._vectorized_system_message,
                model=self.model_name,
                input=Request(user_messages=user_messages).model_dump_json(),
                text_format=ResponseT,
                **self.api_kwargs,
            )
        except BadRequestError as e:
            _handle_temperature_error(e, self.model_name, self.api_kwargs.get("temperature", 0.0))
            raise  # Re-raise if it wasn't a temperature error

        return cast(ParsedResponse[Response[ResponseFormat]], response)

    @observe(_LOGGER)
    def _predict_chunk(self, user_messages: list[str]) -> list[ResponseFormat | None]:
        """Helper executed for every unique minibatch.

            This method:
            1. Converts plain strings into `Message[str]` with stable indices.
            2. Delegates the request to `_request_llm`.
            3. Reorders the responses so they match the original indices.

        The function is pure – it has no side‑effects and the result depends
        only on its arguments – which allows safe reuse.
        """
        messages = [Message(id=i, body=message) for i, message in enumerate(user_messages)]
        responses: ParsedResponse[Response[ResponseFormat]] = self._request_llm(messages)
        if not responses.output_parsed:
            return [None] * len(messages)
        response_dict = {message.id: message.body for message in responses.output_parsed.assistant_messages}
        sorted_responses: list[ResponseFormat | None] = [response_dict.get(m.id, None) for m in messages]
        return sorted_responses

    @observe(_LOGGER)
    def parse(self, inputs: list[str]) -> list[ResponseFormat | None]:
        """Batched predict.

        Args:
            inputs (list[str]): Prompts that require responses. Duplicates are de‑duplicated.

        Returns:
            list[ResponseFormat | None]: Assistant responses aligned to ``inputs``.
        """
        return self.cache.map(inputs, self._predict_chunk)  # type: ignore[return-value]


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
        response_format (type[ResponseFormat]): Expected Pydantic model class or ``str`` for each assistant message.
        cache (AsyncBatchingMapProxy[str, ResponseFormat]): Async batching proxy with de‑duplication
            and concurrency control.
    """

    client: AsyncOpenAI
    model_name: str  # For Azure: deployment name, for OpenAI: model name
    system_message: str
    response_format: type[ResponseFormat] = str  # type: ignore[assignment]
    cache: AsyncBatchingMapProxy[str, ResponseFormat] = field(
        default_factory=lambda: AsyncBatchingMapProxy(batch_size=None, max_concurrency=8)
    )
    api_kwargs: dict[str, int | float | str | bool] = field(default_factory=dict)
    _vectorized_system_message: str = field(init=False)
    _model_json_schema: dict = field(init=False)

    @classmethod
    def of(
        cls,
        client: AsyncOpenAI,
        model_name: str,
        system_message: str,
        response_format: type[ResponseFormat] = str,
        batch_size: int | None = None,
        max_concurrency: int = 8,
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
            **api_kwargs: Additional OpenAI API parameters (temperature, top_p, etc.).

        Returns:
            AsyncBatchResponses: Configured instance backed by an async batching proxy.
        """
        return cls(
            client=client,
            model_name=model_name,
            system_message=system_message,
            response_format=response_format,
            cache=AsyncBatchingMapProxy(batch_size=batch_size, max_concurrency=max_concurrency),
            api_kwargs=api_kwargs,
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
            response_format=task.response_format,
            cache=AsyncBatchingMapProxy(batch_size=batch_size, max_concurrency=max_concurrency),
            api_kwargs=task.api_kwargs,
        )

    def __post_init__(self):
        object.__setattr__(
            self,
            "_vectorized_system_message",
            _vectorize_system_message(self.system_message),
        )

    @backoff_async(exceptions=[RateLimitError, InternalServerError], scale=1, max_retries=12)
    @observe(_LOGGER)
    async def _request_llm(self, user_messages: list[Message[str]]) -> ParsedResponse[Response[ResponseFormat]]:
        """Make a single async call to the OpenAI JSON‑mode endpoint.

        Args:
            user_messages (list[Message[str]]): Sequence of ``Message[str]`` representing the minibatch prompts.

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
            assistant_messages: list[MessageT]

        try:
            response: ParsedResponse[ResponseT] = await self.client.responses.parse(
                instructions=self._vectorized_system_message,
                model=self.model_name,
                input=Request(user_messages=user_messages).model_dump_json(),
                text_format=ResponseT,
                **self.api_kwargs,
            )
        except BadRequestError as e:
            _handle_temperature_error(e, self.model_name, self.api_kwargs.get("temperature", 0.0))
            raise  # Re-raise if it wasn't a temperature error

        return cast(ParsedResponse[Response[ResponseFormat]], response)

    @observe(_LOGGER)
    async def _predict_chunk(self, user_messages: list[str]) -> list[ResponseFormat | None]:
        """Async helper executed for every unique minibatch.

            This method:
            1. Converts plain strings into `Message[str]` with stable indices.
            2. Delegates the request to `_request_llm`.
            3. Reorders the responses so they match the original indices.

        The function is pure – it has no side‑effects and the result depends only on its arguments.
        """
        messages = [Message(id=i, body=message) for i, message in enumerate(user_messages)]
        responses: ParsedResponse[Response[ResponseFormat]] = await self._request_llm(messages)
        if not responses.output_parsed:
            return [None] * len(messages)
        response_dict = {message.id: message.body for message in responses.output_parsed.assistant_messages}
        # Ensure proper handling for missing IDs - this shouldn't happen in normal operation
        sorted_responses: list[ResponseFormat | None] = [response_dict.get(m.id, None) for m in messages]
        return sorted_responses

    @observe(_LOGGER)
    async def parse(self, inputs: list[str]) -> list[ResponseFormat | None]:
        """Batched predict (async).

        Args:
            inputs (list[str]): Prompts that require responses. Duplicates are de‑duplicated.

        Returns:
            list[ResponseFormat | None]: Assistant responses aligned to ``inputs``.
        """
        return await self.cache.map(inputs, self._predict_chunk)  # type: ignore[return-value]
