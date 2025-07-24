import asyncio
from dataclasses import dataclass, field
from logging import Logger, getLogger
from typing import Generic, List, Type, TypeVar, cast

from openai import AsyncOpenAI, OpenAI, RateLimitError
from openai.types.responses import ParsedResponse
from pydantic import BaseModel

from .log import observe
from .task.model import PreparedTask
from .util import backoff, backoff_async, map, map_async

__all__ = [
    "BatchResponses",
    "AsyncBatchResponses",
]

_LOGGER: Logger = getLogger(__name__)


def _vectorize_system_message(system_message: str) -> str:
    """Return the system prompt that instructs the model to work on a batch.

    The returned XML‐ish prompt explains two things to the LLM:

    1. The *general* system instruction coming from the caller (`system_message`)
       is preserved verbatim.
    2. Extra instructions describe how the model should treat the incoming JSON
       that contains multiple user messages and how it must shape its output.

    Args:
        system_message: A single‑instance system instruction the caller would
            normally send to the model.

    Returns:
        A long, composite system prompt with embedded examples that can be
        supplied to the `instructions=` field of the OpenAI **JSON mode**
        endpoint.
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


T = TypeVar("T")


class Message(BaseModel, Generic[T]):
    id: int
    body: T


class Request(BaseModel):
    user_messages: List[Message[str]]


class Response(BaseModel, Generic[T]):
    assistant_messages: List[Message[T]]


@dataclass(frozen=True)
class BatchResponses(Generic[T]):
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
        answers = vector_llm.parse(questions, batch_size=32)
        ```

    Attributes:
        client: Initialised ``openai.OpenAI`` client.
        model_name: Name of the model (or Azure deployment) to invoke.
        system_message: System prompt prepended to every request.
        temperature: Sampling temperature passed to the model.
        top_p: Nucleus‑sampling parameter.
        response_format: Expected Pydantic type of each assistant message
            (defaults to ``str``).

    Notes:
        Internally the work is delegated to two helpers:

        * ``_predict_chunk`` – fragments the workload and restores ordering.
        * ``_request_llm`` – performs a single OpenAI API call.
    """

    client: OpenAI
    model_name: str  # it would be the name of deployment for Azure
    system_message: str
    temperature: float = 0.0
    top_p: float = 1.0
    response_format: Type[T] = str
    _vectorized_system_message: str = field(init=False)
    _model_json_schema: dict = field(init=False)

    @classmethod
    def of_task(cls, client: OpenAI, model_name: str, task: PreparedTask) -> "BatchResponses":
        """Create a BatchResponses instance from a PreparedTask."""
        return cls(
            client=client,
            model_name=model_name,
            system_message=task.instructions,
            temperature=task.temperature,
            top_p=task.top_p,
            response_format=task.response_format,
        )

    def __post_init__(self):
        object.__setattr__(
            self,
            "_vectorized_system_message",
            _vectorize_system_message(self.system_message),
        )

    @observe(_LOGGER)
    @backoff(exception=RateLimitError, scale=15, max_retries=8)
    def _request_llm(self, user_messages: List[Message[str]]) -> ParsedResponse[Response[T]]:
        """Make a single call to the OpenAI *JSON mode* endpoint.

        Args:
            user_messages: Sequence of `Message[str]` objects representing the
                prompts for this minibatch.  Each message carries a unique `id`
                so we can restore ordering later.

        Returns:
            ParsedResponse containing `Response[T]` which in turn holds the
            assistant messages in arbitrary order.

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

        completion: ParsedResponse[ResponseT] = self.client.responses.parse(
            model=self.model_name,
            instructions=self._vectorized_system_message,
            input=Request(user_messages=user_messages).model_dump_json(),
            temperature=self.temperature,
            top_p=self.top_p,
            text_format=ResponseT,
        )
        return cast(ParsedResponse[Response[T]], completion)

    @observe(_LOGGER)
    def _predict_chunk(self, user_messages: List[str]) -> List[T]:
        """Helper executed for every unique minibatch.

        This method:
        1. Converts plain strings into `Message[str]` with stable indices.
        2. Delegates the request to `_request_llm`.
        3. Reorders the responses so they match the original indices.

        The function is *pure* – it has no side‑effects and the result depends
        only on its arguments – which allows it to be used safely in both
        serial and parallel execution paths.
        """
        messages = [Message(id=i, body=message) for i, message in enumerate(user_messages)]
        responses: ParsedResponse[Response[T]] = self._request_llm(messages)
        response_dict = {message.id: message.body for message in responses.output_parsed.assistant_messages}
        sorted_responses = [response_dict.get(m.id, None) for m in messages]
        return sorted_responses

    @observe(_LOGGER)
    def parse(self, inputs: List[str], batch_size: int) -> List[T]:
        """Public API: batched predict.

        Args:
            inputs: All prompts that require a response.  Duplicate
                entries are de‑duplicated under the hood to save tokens.
            batch_size: Maximum number of *unique* prompts per LLM call.

        Returns:
            A list containing the assistant responses in the same order as
                *inputs*.
        """
        return map(inputs, self._predict_chunk, batch_size)


@dataclass(frozen=True)
class AsyncBatchResponses(Generic[T]):
    """Stateless façade that turns OpenAI's JSON-mode API into a batched API (Async version).

    This wrapper allows you to submit *multiple* user prompts in one JSON-mode
    request and receive the answers in the original order asynchronously. It also
    controls the maximum number of concurrent requests to the OpenAI API.

    Example:
        ```python
        import asyncio
        from openai import AsyncOpenAI
        from openaivec.aio.responses import AsyncBatchResponses

        # Assuming openai_async_client is an initialized AsyncOpenAI client
        openai_async_client = AsyncOpenAI() # Replace with your actual client initialization

        vector_llm = AsyncBatchResponses(
            client=openai_async_client,
            model_name="gpt-4o-mini",
            system_message="You are a helpful assistant.",
            max_concurrency=5  # Limit concurrent requests
        )
        questions = ["What is the capital of France?", "Explain quantum physics simply."]
        # Asynchronous call
        async def main():
            answers = await vector_llm.parse(questions, batch_size=32)
            print(answers)

        # Run the async function
        asyncio.run(main())
        ```

    Attributes:
        client: Initialised `openai.AsyncOpenAI` client.
        model_name: Name of the model (or Azure deployment) to invoke.
        system_message: System prompt prepended to every request.
        temperature: Sampling temperature passed to the model.
        top_p: Nucleus-sampling parameter.
        response_format: Expected Pydantic type of each assistant message
            (defaults to `str`).
        max_concurrency: Maximum number of concurrent requests to the OpenAI API.
    """

    client: AsyncOpenAI
    model_name: str  # it would be the name of deployment for Azure
    system_message: str
    temperature: float = 0.0
    top_p: float = 1.0
    response_format: Type[T] = str
    max_concurrency: int = 8  # Default concurrency limit
    _vectorized_system_message: str = field(init=False)
    _model_json_schema: dict = field(init=False)
    _semaphore: asyncio.Semaphore = field(init=False, repr=False)

    @classmethod
    def of_task(cls, client: AsyncOpenAI, model_name: str, task: PreparedTask, max_concurrency: int = 8) -> "AsyncBatchResponses":
        """Create an AsyncBatchResponses instance from a PreparedTask."""
        return cls(
            client=client,
            model_name=model_name,
            system_message=task.instructions,
            temperature=task.temperature,
            top_p=task.top_p,
            response_format=task.response_format,
            max_concurrency=max_concurrency,
        )

    def __post_init__(self):
        object.__setattr__(
            self,
            "_vectorized_system_message",
            _vectorize_system_message(self.system_message),
        )
        # Initialize the semaphore after the object is created
        # Use object.__setattr__ because the dataclass is frozen
        object.__setattr__(self, "_semaphore", asyncio.Semaphore(self.max_concurrency))

    @observe(_LOGGER)
    @backoff_async(exception=RateLimitError, scale=15, max_retries=8)
    async def _request_llm(self, user_messages: List[Message[str]]) -> ParsedResponse[Response[T]]:
        """Make a single async call to the OpenAI *JSON mode* endpoint, respecting concurrency limits.

        Args:
            user_messages: Sequence of `Message[str]` objects representing the
                prompts for this minibatch. Each message carries a unique `id`
                so we can restore ordering later.

        Returns:
            ParsedResponse containing `Response[T]` which in turn holds the
            assistant messages in arbitrary order.

        Raises:
            openai.RateLimitError: Transparently re-raised after the
                exponential back-off decorator exhausts all retries.
        """
        response_format = self.response_format

        class MessageT(BaseModel):
            id: int
            body: response_format  # type: ignore

        class ResponseT(BaseModel):
            assistant_messages: List[MessageT]

        # Acquire semaphore before making the API call
        async with self._semaphore:
            # Directly await the async call instead of using asyncio.run()
            completion: ParsedResponse[ResponseT] = await self.client.responses.parse(
                model=self.model_name,
                instructions=self._vectorized_system_message,
                input=Request(user_messages=user_messages).model_dump_json(),
                temperature=self.temperature,
                top_p=self.top_p,
                text_format=ResponseT,
            )
            return cast(ParsedResponse[Response[T]], completion)

    @observe(_LOGGER)
    async def _predict_chunk(self, user_messages: List[str]) -> List[T]:
        """Helper executed asynchronously for every unique minibatch.

        This method:
        1. Converts plain strings into `Message[str]` with stable indices.
        2. Delegates the request to `_request_llm`.
        3. Reorders the responses so they match the original indices.

        The function is *pure* – it has no side-effects and the result depends
        only on its arguments.
        """
        messages = [Message(id=i, body=message) for i, message in enumerate(user_messages)]
        responses: ParsedResponse[Response[T]] = await self._request_llm(messages)
        response_dict = {message.id: message.body for message in responses.output_parsed.assistant_messages}
        # Ensure proper handling for missing IDs - this shouldn't happen in normal operation
        sorted_responses = [response_dict.get(m.id, None) for m in messages]
        return sorted_responses

    @observe(_LOGGER)
    async def parse(self, inputs: List[str], batch_size: int) -> List[T]:
        """Asynchronous public API: batched predict.

        Args:
            inputs: All prompts that require a response. Duplicate
                entries are de-duplicated under the hood to save tokens.
            batch_size: Maximum number of *unique* prompts per LLM call.

        Returns:
            A list containing the assistant responses in the same order as
                *inputs*.
        """

        return await map_async(
            inputs=inputs,
            f=self._predict_chunk,
            batch_size=batch_size,  # Use the batch_size argument passed to the method
        )
