"""Task model module for openaivec.

This module provides predefined task models for the OpenAI API.
The `PreparedTask` class represents a complete task configuration including
instructions, response format, and sampling parameters.

Example:
    Basic usage with a predefined task:
    
    ```python
    from openai import OpenAI
    from openaivec.task import nlp
    from openaivec.responses import BatchResponses

    translation_task: BatchResponses = BatchResponses.of_task(
        client=OpenAI(),
        model_name="gpt-4.1-mini",
        task=nlp.MULTILINGUAL_TRANSLATION
    )
    ```

Note:
    This module uses Pydantic models for response format validation and
    type safety.
"""

from dataclasses import dataclass
from typing import Type, TypeVar
from pydantic import BaseModel

__all__ = ['PreparedTask']

T = TypeVar('T', bound=BaseModel)



@dataclass(frozen=True)
class PreparedTask:
    """A data class representing a complete task configuration for OpenAI API calls.
    
    This class encapsulates all the necessary parameters for executing a task,
    including the instructions to be sent to the model, the expected response
    format using Pydantic models, and sampling parameters for controlling
    the model's output behavior.
    
    Attributes:
        instructions (str): The prompt or instructions to send to the OpenAI model.
            This should contain clear, specific directions for the task.
        response_format (Type[T]): A Pydantic model class that defines the expected
            structure of the response. Must inherit from BaseModel.
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
    response_format: Type[T]
    temperature: float = 0.0
    top_p: float = 1.0
