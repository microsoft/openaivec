from dataclasses import dataclass
from typing import Generic, Type, TypeVar

__all__ = [
    "PreparedTask",
]

ResponseFormat = TypeVar("ResponseFormat")


@dataclass(frozen=True)
class PreparedTask(Generic[ResponseFormat]):
    """A data class representing a complete task configuration for OpenAI API calls.

    This class encapsulates all the necessary parameters for executing a task,
    including the instructions to be sent to the model, the expected response
    format using Pydantic models, and sampling parameters for controlling
    the model's output behavior.

    Attributes:
        instructions (str): The prompt or instructions to send to the OpenAI model.
            This should contain clear, specific directions for the task.
        response_format (Type[ResponseFormat]): A Pydantic model class or str type that defines the expected
            structure of the response. Can be either a BaseModel subclass or str.
        temperature (float): Controls randomness in the model's output.
            Range: 0.0 to 1.0. Lower values make output more deterministic.
            Defaults to 0.0.
        top_p (float): Controls diversity via nucleus sampling. Only tokens
            comprising the top_p probability mass are considered.
            Range: 0.0 to 1.0. Defaults to 1.0.

    Example:
        Creating a custom task:

        ```python
        from pydantic import BaseModel

        class TranslationResponse(BaseModel):
            translated_text: str
            source_language: str
            target_language: str

        custom_task = PreparedTask(
            instructions="Translate the following text to French:",
            response_format=TranslationResponse,
            temperature=0.1,
            top_p=0.9
        )
        ```

    Note:
        This class is frozen (immutable) to ensure task configurations
        cannot be accidentally modified after creation.
    """

    instructions: str
    response_format: Type[ResponseFormat]
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass(frozen=True)
class ResponsesModelName:
    """Container for responses model name configuration.

    Attributes:
        value (str): The model name for OpenAI responses API.
    """

    value: str


@dataclass(frozen=True)
class EmbeddingsModelName:
    """Container for embeddings model name configuration.

    Attributes:
        value (str): The model name for OpenAI embeddings API.
    """

    value: str


@dataclass(frozen=True)
class OpenAIAPIKey:
    """Container for OpenAI API key configuration.

    Attributes:
        value (str | None): The API key for OpenAI services.
    """

    value: str | None


@dataclass(frozen=True)
class AzureOpenAIAPIKey:
    """Container for Azure OpenAI API key configuration.

    Attributes:
        value (str | None): The API key for Azure OpenAI services.
    """

    value: str | None


@dataclass(frozen=True)
class AzureOpenAIBaseURL:
    """Container for Azure OpenAI base URL configuration.

    Attributes:
        value (str | None): The base URL for Azure OpenAI services.
    """

    value: str | None


@dataclass(frozen=True)
class AzureOpenAIAPIVersion:
    """Container for Azure OpenAI API version configuration.

    Attributes:
        value (str): The API version for Azure OpenAI services.
    """

    value: str
