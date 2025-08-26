"""
This module provides a builder for creating few‑shot prompts, which are
used to train large language models (LLMs) by providing them with
examples of input/output pairs. The builder allows for the
construction of a prompt in a structured way, including setting the
purpose, adding cautions, and providing examples.

```python
from openaivec.prompt import FewShotPromptBuilder

prompt_str: str = (
    FewShotPromptBuilder()
    .purpose("some purpose")
    .caution("some caution")
    .caution("some other caution")
    .example("some input", "some output")
    .example("some other input", "some other output")
    .build()
)
print(prompt_str)
```
this will produce an XML string that looks like this:
```xml
<Prompt>
    <Purpose>some purpose</Purpose>
    <Cautions>
        <Caution>some caution</Caution>
        <Caution>some other caution</Caution>
    </Cautions>
    <Examples>
        <Example>
            <Input>some input</Input>
            <Output>some output</Output>
        </Example>
        <Example>
            <Input>some other input</Input>
            <Output>some other output</Output>
        </Example>
    </Examples>
</Prompt>
```

"""

import difflib
import logging
from xml.etree import ElementTree

from openai import OpenAI
from openai.types.responses import ParsedResponse
from pydantic import BaseModel

from openaivec._model import ResponsesModelName
from openaivec._provider import CONTAINER

__all__ = [
    "FewShotPrompt",
    "FewShotPromptBuilder",
]

_logger = logging.getLogger(__name__)


class Example(BaseModel):
    """Represents a single input/output example used in few‑shot prompts.

    Attributes:
        input (str): The input text that will be passed to the model.
        output (str): The expected output corresponding to the given input.
    """

    input: str
    output: str


class FewShotPrompt(BaseModel):
    """Represents a prompt definition used for few‑shot learning.

    The data collected in this model is later rendered into XML and sent to a
    large‑language model as part of the system prompt.

    Attributes:
        purpose (str): A concise, human‑readable statement describing the goal
            of the prompt.
        cautions (list[str]): A list of warnings, edge cases, or pitfalls that
            the model should be aware of when generating answers.
        examples (list[Example]): Input/output pairs demonstrating the expected
            behaviour for a variety of scenarios.
    """

    purpose: str
    cautions: list[str]
    examples: list[Example]


class Step(BaseModel):
    """A single refinement iteration produced by the LLM.

    Attributes:
        id (int): Sequential identifier of the iteration (``0`` for the
            original, ``1`` for the first change, and so on).
        analysis (str): Natural‑language explanation of the issue addressed
            in this iteration and why the change was necessary.
        prompt (FewShotPrompt): The updated prompt after applying the
            described modification.
    """

    id: int
    analysis: str
    prompt: FewShotPrompt


class Request(BaseModel):
    prompt: FewShotPrompt


class Response(BaseModel):
    iterations: list[Step]


_PROMPT: str = """
<Prompt>
    <Instructions>
        <Instruction id="1">
            Receive the prompt in JSON format with fields "purpose",
            "cautions", and "examples". Ensure the entire prompt is free
            from logical contradictions, redundancies, and ambiguities.
            IMPORTANT: The "examples" array must always contain at least one example throughout all iterations.
        </Instruction>
        <Instruction id="2">
            - Modify only one element per iteration among “purpose”, “examples”, or
              “cautions”, refining each at least once.
            - Address exactly one type of issue in each step.
            - Focus solely on that issue and provide a detailed explanation of the
              problem and its negative impacts.
            - Append the results sequentially to the ‘iterations’ field.
            - Write the explanation in the ‘analysis’ field and the updated prompt in
              the ‘prompt’ field.
            - Continue iterations until all issues have been addressed.
            - For the final step, review the entire prompt to ensure no issues remain
              and apply any necessary modifications.
        </Instruction>
        <Instruction id="3">
            Always respond in the same language as specified in the "purpose" field for all output values,
            including the analysis field and chain-of-thought steps.
        </Instruction>
        <Instruction id="4">
            In the "purpose" field, clearly describe the overall semantics and main goal,
            ensuring that all critical instructions contained in the original text are
            preserved without altering the base meaning.
        </Instruction>
        <Instruction id="5">
            In the "cautions" field, list common points or edge cases found
            in the examples.
        </Instruction>
        <Instruction id="6">
            In the "examples" field, enhance the examples to cover a wide range of scenarios.
            CRITICAL: The examples array must NEVER be empty - always maintain at least one example.
            Add as many non-redundant examples as possible,
            since having more examples leads to better coverage and understanding.
            You may modify existing examples or add new ones, but never remove all examples.
        </Instruction>
        <Instruction id="7">
            Verify that the improved prompt adheres to the Request and
            Response JSON schemas.
        </Instruction>
        <Instruction id="8">
            Generate the final refined FewShotPrompt as an iteration in
            the Response, ensuring the final output is consistent,
            unambiguous, and free from any redundancies or contradictions.
            MANDATORY: Verify that the examples array contains at least one example before completing.
        </Instruction>
    </Instructions>
    <Example>
        <Input>{
    "origin": {
        "purpose": "some_purpose01",
        "cautions": [
            "some_caution01",
            "some_caution02",
            "some_caution03"
        ],
        "examples": [
            {
                "input": "some_input01",
                "output": "some_output01"
            },
            {
                "input": "some_input02",
                "output": "some_output02"
            },
            {
                "input": "some_input03",
                "output": "some_output03"
            },
            {
                "input": "some_input04",
                "output": "some_output04"
            },
            {
                "input": "some_input05",
                "output": "some_output05"
            }
        ]
    }
}</Input>
<Output>
{
  "iterations": [
    {
      "id": 1,
      "analysis": "The original purpose was vague and did not explicitly state the main objective.
        This ambiguity could lead to confusion about the task. In this iteration, we refined the purpose to
        clearly specify that the goal is to determine the correct category for a given word based on its context.",
      "prompt": {
        "purpose": "Determine the correct category for a given word by analyzing its context for clear meaning.",
        "cautions": [
          "Ensure the word's context is provided to avoid ambiguity.",
          "Consider multiple meanings of the word and choose the most relevant category."
        ],
        "examples": [
          {
            "input": "Apple (as a fruit)",
            "output": "Fruit"
          },
          {
            "input": "Apple (as a tech company)",
            "output": "Technology"
          },
          ...
        ]
      }
    },
    {
      "id": 2,
      "analysis": "Next, we focused solely on the cautions section. The original cautions were generic and
        did not mention potential pitfalls like homonyms or polysemy. Failing to address these could result in
        misclassification. Therefore, we added a specific caution regarding homonyms while keeping the purpose
        and examples unchanged.",
      "prompt": {
        "purpose": "Determine the correct category for a given word by analyzing its context for clear meaning.",
        "cautions": [
          "Ensure the word's context is provided to avoid ambiguity.",
          "Consider multiple meanings of the word and choose the most relevant category.",
          "Pay close attention to homonyms and polysemy to prevent misclassification."
        ],
        "examples": [
          {
            "input": "Apple (as a fruit)",
            "output": "Fruit"
          },
          {
            "input": "Apple (as a tech company)",
            "output": "Technology"
          },
          ...
        ]
      }
    },
    {
      "id": 3,
      "analysis": "In this step, we improved the examples section to cover a broader range of scenarios and
        address potential ambiguities. By adding examples that include words with multiple interpretations
        (such as 'Mercury' for both a planet and an element), we enhance clarity and ensure better coverage.
        This iteration only modifies the examples section, leaving purpose and cautions intact.",
      "prompt": {
        "purpose": "Determine the correct category for a given word by analyzing its context for clear meaning.",
        "cautions": [
          "Ensure the word's context is provided to avoid ambiguity.",
          "Consider multiple meanings of the word and choose the most relevant category.",
          "Pay close attention to homonyms and polysemy to prevent misclassification."
        ],
        "examples": [
          {
            "input": "Apple (as a fruit)",
            "output": "Fruit"
          },
          {
            "input": "Apple (as a tech company)",
            "output": "Technology"
          },
          {
            "input": "Mercury (as a planet)",
            "output": "Astronomy"
          },
          {
            "input": "Mercury (as an element)",
            "output": "Chemistry"
          },
          ...
        ]
      }
    },
    {
        "id": 4,
        "analysis": "In this final iteration, we ensured that the entire prompt...",
        ...
    }
    ...
  ]
}
</Output>
    </Example>
</Prompt>
"""


def _render_prompt(prompt: FewShotPrompt) -> str:
    """Render a FewShotPrompt instance to its XML representation.

    Args:
        prompt (FewShotPrompt): The prompt object to render.

    Returns:
        str: The XML string representation of the prompt.
    """
    prompt_dict = prompt.model_dump()
    root = ElementTree.Element("Prompt")

    # Purpose (always output)
    purpose_elem = ElementTree.SubElement(root, "Purpose")
    purpose_elem.text = prompt_dict["purpose"]

    # Cautions (always output, even if empty)
    cautions_elem = ElementTree.SubElement(root, "Cautions")
    if prompt_dict.get("cautions"):
        for caution in prompt_dict["cautions"]:
            caution_elem = ElementTree.SubElement(cautions_elem, "Caution")
            caution_elem.text = caution

    # Examples (always output)
    examples_elem = ElementTree.SubElement(root, "Examples")
    for example in prompt_dict["examples"]:
        example_elem = ElementTree.SubElement(examples_elem, "Example")
        input_elem = ElementTree.SubElement(example_elem, "Input")
        input_elem.text = example.get("input")
        output_elem = ElementTree.SubElement(example_elem, "Output")
        output_elem.text = example.get("output")

    ElementTree.indent(root, level=0)
    return ElementTree.tostring(root, encoding="unicode")


class FewShotPromptBuilder:
    """Builder for creating few-shot prompts with validation.

    Usage:
        builder = (FewShotPromptBuilder()
                  .purpose("Your task description")
                  .example("input1", "output1")  # At least one required
                  .example("input2", "output2")
                  .build())

    Note:
        Both .purpose() and at least one .example() call are required before
        calling .build(), .improve(), or .get_object().
    """

    _prompt: FewShotPrompt
    _steps: list[Step]

    def __init__(self):
        """Initialize an empty FewShotPromptBuilder.

        Note:
            You must call .purpose() and at least one .example() before building.
        """
        self._prompt = FewShotPrompt(purpose="", cautions=[], examples=[])

    @classmethod
    def of(cls, prompt: FewShotPrompt) -> "FewShotPromptBuilder":
        """Create a builder pre‑populated with an existing prompt.

        Args:
            prompt (FewShotPrompt): The prompt to start from.

        Returns:
            FewShotPromptBuilder: A new builder instance.
        """
        builder = cls()
        builder._prompt = prompt
        return builder

    @classmethod
    def of_empty(cls) -> "FewShotPromptBuilder":
        """Create a builder.

        Returns:
            FewShotPromptBuilder: A new builder instance with an empty prompt.
        """
        return cls.of(FewShotPrompt(purpose="", cautions=[], examples=[]))

    def purpose(self, purpose: str) -> "FewShotPromptBuilder":
        """Set the purpose of the prompt.

        Args:
            purpose (str): A concise statement describing the prompt’s goal.

        Returns:
            FewShotPromptBuilder: The current builder instance (for chaining).
        """
        self._prompt.purpose = purpose
        return self

    def caution(self, caution: str) -> "FewShotPromptBuilder":
        """Append a cautionary note to the prompt.

        Args:
            caution (str): A caution or edge‑case description.

        Returns:
            FewShotPromptBuilder: The current builder instance.
        """
        if self._prompt.cautions is None:
            self._prompt.cautions = []
        self._prompt.cautions.append(caution)
        return self

    def example(
        self,
        input_value: str | BaseModel,
        output_value: str | BaseModel,
    ) -> "FewShotPromptBuilder":
        """Add a single input/output example.

        At least one example is required before calling .build(), .improve(), or .get_object().

        Args:
            input_value (str | BaseModel): Example input; if a Pydantic model is
                provided it is serialised to JSON.
            output_value (str | BaseModel): Expected output; serialised if needed.

        Returns:
            FewShotPromptBuilder: The current builder instance.
        """
        if self._prompt.examples is None:
            self._prompt.examples = []

        input_string = input_value if isinstance(input_value, str) else input_value.model_dump_json()
        output_string = output_value if isinstance(output_value, str) else output_value.model_dump_json()
        self._prompt.examples.append(Example(input=input_string, output=output_string))
        return self

    def improve(
        self,
        client: OpenAI | None = None,
        model_name: str | None = None,
        **api_kwargs,
    ) -> "FewShotPromptBuilder":
        """Iteratively refine the prompt using an LLM.

        The method calls a single LLM request that returns multiple
        editing steps and stores each step for inspection.

        When client is None, automatically creates a client using environment variables:
        - For OpenAI: ``OPENAI_API_KEY``
        - For Azure OpenAI: ``AZURE_OPENAI_API_KEY``, ``AZURE_OPENAI_BASE_URL``, ``AZURE_OPENAI_API_VERSION``

        Args:
            client (OpenAI | None): Configured OpenAI client. If None, uses DI container with environment variables.
            model_name (str | None): Model identifier. If None, uses default ``gpt-4.1-mini``.
            **api_kwargs: Additional OpenAI API parameters (temperature, top_p, etc.).

        Returns:
            FewShotPromptBuilder: The current builder instance containing the refined prompt and iteration history.

        Raises:
            ValueError: If the prompt is not valid (missing purpose or examples).
        """
        # Validate before making API call to provide early feedback
        self._validate()

        _client = client or CONTAINER.resolve(OpenAI)
        _model_name = model_name or CONTAINER.resolve(ResponsesModelName).value

        response: ParsedResponse[Response] = _client.responses.parse(
            model=_model_name,
            instructions=_PROMPT,
            input=Request(prompt=self._prompt).model_dump_json(),
            text_format=Response,
            **api_kwargs,
        )

        # keep the original prompt
        self._steps = [Step(id=0, analysis="Original Prompt", prompt=self._prompt)]

        # add the histories
        if response.output_parsed:
            for step in response.output_parsed.iterations:
                self._steps.append(step)

        # set the final prompt
        self._prompt = self._steps[-1].prompt

        # Validate the improved prompt to ensure examples weren't removed by LLM
        try:
            self._validate()
        except ValueError as e:
            _logger.warning(f"LLM produced invalid prompt during improve(): {e}")
            # Restore original prompt if LLM produced invalid result
            self._prompt = self._steps[0].prompt
            raise ValueError(
                f"LLM improvement failed to maintain required fields: {e}. "
                "This may indicate an issue with the improvement instructions or model behavior."
            )

        return self

    def explain(self) -> "FewShotPromptBuilder":
        """Pretty‑print the diff of each improvement iteration.

        Returns:
            FewShotPromptBuilder: The current builder instance.
        """
        if not hasattr(self, "_steps") or not self._steps:
            print("No improvement steps available. Call improve() first.")
            return self

        for previous, current in zip(self._steps, self._steps[1:]):
            print(f"=== Iteration {current.id} ===\n")
            print(f"Instruction: {current.analysis}")
            diff = difflib.unified_diff(
                _render_prompt(previous.prompt).splitlines(),
                _render_prompt(current.prompt).splitlines(),
                fromfile="before",
                tofile="after",
                lineterm="",
            )
            for line in diff:
                print(line)
        return self

    def _validate(self) -> None:
        """Validate the internal FewShotPrompt.

        Raises:
            ValueError: If required fields such as purpose or examples are
                missing.
        """
        # Validate that 'purpose' and 'examples' are not empty.
        if not self._prompt.purpose:
            raise ValueError(
                "Purpose is required. Please call .purpose('your purpose description') before building the prompt."
            )
        if not self._prompt.examples or len(self._prompt.examples) == 0:
            raise ValueError(
                "At least one example is required. Please add examples using "
                ".example('input', 'output') before building the prompt."
            )

    def get_object(self) -> FewShotPrompt:
        """Return the underlying FewShotPrompt object.

        Returns:
            FewShotPrompt: The validated prompt object.
        """
        self._validate()
        return self._prompt

    def build(self) -> str:
        """Build and return the prompt as XML.

        Returns:
            str: XML representation of the prompt.
        """
        self._validate()
        return self.build_xml()

    def build_json(self, **kwargs) -> str:
        """Build and return the prompt as a JSON string.

        Args:
            **kwargs: Keyword arguments forwarded to Pydantic's ``model_dump_json``.
                Common options include ``indent``, ``include``, ``exclude``,
                ``by_alias``, ``exclude_unset``, ``exclude_defaults``, ``exclude_none``.

        Returns:
            str: JSON representation of the prompt.
        """
        self._validate()
        return self._prompt.model_dump_json(**kwargs)

    def build_xml(self) -> str:
        """Alias for :py:meth:`build` for explicit XML generation.

        Returns:
            str: XML representation of the prompt.
        """
        self._validate()
        return _render_prompt(self._prompt)
