from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Optional, Type

from openai import OpenAI
from openai.types.responses import ParsedResponse
from pydantic import BaseModel, Field, create_model

from openaivec._model import PreparedTask

# Internal module: explicitly not part of public API
__all__: list[str] = []


class FieldSpec(BaseModel):
    name: str = Field(
        description=(
            "Lower snake_case identifier (regex: ^[a-z][a-z0-9_]*$). Must be unique across all fields and "
            "express the semantic meaning succinctly (no adjectives like 'best', 'great')."
        )
    )
    type: Literal["string", "integer", "float", "boolean"] = Field(
        description=(
            "Primitive type. Use 'integer' only if all observed numeric values are whole numbers. "
            "Use 'float' if any value can contain a decimal or represents a ratio/score. Use 'boolean' only for "
            "explicit binary states (yes/no, true/false, present/absent) consistently encoded. Use 'string' otherwise. "
            "Never output arrays, objects, or composite encodings; flatten to the most specific scalar value."
        )
    )
    description: str = Field(
        description=(
            "Concise, objective definition plus extraction rule (what qualifies / what to ignore). Avoid subjective, "
            "speculative, or promotional language. If ambiguity exists with another field, clarify the distinction."
        )
    )
    enum_values: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional finite categorical label set (classification) for a string field. Provide ONLY when a closed, "
            "stable vocabulary (2–24 lowercase tokens) is clearly evidenced or strongly implied by examples. "
            "Do NOT invent labels. Omit if open-ended or ambiguous. Order must be stable and semantically natural."
        ),
    )


class InferredSchema(BaseModel):
    purpose: str = Field(
        description=(
            "Normalized, unambiguous restatement of the user objective with redundant, vague, or "
            "conflicting phrasing removed."
        )
    )
    examples_summary: str = Field(
        description=(
            "Objective characterization of the provided examples: content domain, structure, recurring "
            "patterns, and notable constraints."
        )
    )
    fields: List[FieldSpec] = Field(
        description=(
            "Ordered list of proposed fields derived strictly from observable, repeatable signals in the "
            "examples and aligned with the purpose."
        )
    )
    inference_prompt: str = Field(
        description=(
            "Canonical, reusable extraction prompt for structuring future inputs with this schema. "
            "Must be fully derivable from 'purpose', 'examples_summary', and 'fields' (no new unstated facts or "
            "speculation). It MUST: (1) instruct the model to output only the listed fields with the exact names "
            "and primitive types; (2) forbid adding, removing, or renaming fields; (3) avoid subjective or "
            "marketing language; (4) be self-contained (no TODOs, no external references, no unresolved "
            "placeholders). Intended for direct reuse as the prompt for deterministic alignment with 'fields'."
        )
    )

    @classmethod
    def load(cls, path: str) -> "InferredSchema":
        """Load an inferred schema from a JSON file.

        Args:
            path (str): File path to load the schema JSON.

        Returns:
            InferredSchema: Loaded schema object.
        """
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())

    @property
    def model(self) -> Type[BaseModel]:
        """Return the Pydantic model type for this inferred schema.

        This is a convenience property that calls ``build_model()`` to create the dynamic model.
        """
        return self.build_model()

    @property
    def task(self) -> PreparedTask:
        return PreparedTask(
            instructions=self.inference_prompt, response_format=self.model, top_p=None, temperature=None
        )

    def build_model(self) -> Type[BaseModel]:
        """Materialize a dynamic ``BaseModel`` matching this inferred schema.

        Rules:
          - Primitive mapping: string->str, integer->int, float->float, boolean->bool
          - If ``enum_values`` present, a dynamic ``Enum`` subclass is created and used as the field type.
          - All fields are required (``...``). Adjust here if optionality is introduced later.

        Returns:
            Type[BaseModel]: Dynamically created Pydantic model whose fields mirror ``self.fields``.
        """
        type_map: dict[str, type] = {"string": str, "integer": int, "float": float, "boolean": bool}
        fields: dict[str, tuple[type, object]] = {}

        for spec in self.fields:
            py_type: type
            if spec.enum_values:
                enum_class_name = "Enum_" + "".join(part.capitalize() for part in spec.name.split("_"))
                members: dict[str, str] = {}
                for raw in spec.enum_values:
                    sanitized = raw.upper().replace("-", "_").replace(" ", "_")
                    if not sanitized or sanitized[0].isdigit():
                        sanitized = f"V_{sanitized}"
                    base = sanitized
                    i = 2
                    while sanitized in members:
                        sanitized = f"{base}_{i}"
                        i += 1
                    members[sanitized] = raw
                enum_cls = Enum(enum_class_name, members)  # type: ignore[arg-type]
                py_type = enum_cls
            else:
                py_type = type_map[spec.type]
            fields[spec.name] = (py_type, ...)

        model = create_model("InferredSchema", **fields)  # type: ignore[call-arg]
        return model

    def save(self, path: str) -> None:
        """Save the inferred schema as a JSON file.

        Args:
            path (str): File path to save the schema JSON.
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))


class SchemaInferenceInput(BaseModel):
    examples: List[str] = Field(
        description=(
            "Representative sample texts (strings). Provide only data the schema should generalize over; "
            "exclude outliers not in scope."
        )
    )
    purpose: str = Field(
        description=(
            "Plain language statement describing the downstream use of the extracted structured data (e.g. "
            "analytics, filtering, enrichment)."
        )
    )


_INFER_INSTRUCTIONS = """
You are a schema inference engine.

Task:
1. Normalize the user's purpose (eliminate ambiguity, redundancy, contradictions).
2. Objectively summarize observable patterns in the example texts.
3. Propose a minimal flat set of scalar fields (no nesting / arrays) that are reliably extractable.
4. Skip fields likely missing in a large share (>~20%) of realistic inputs.
5. Provide enum_values ONLY when a small stable closed categorical set (2–24 lowercase tokens)
    is clearly evidenced; never invent.
6. If the purpose indicates prediction (predict / probability / likelihood), output only
    explanatory features (no target restatement).

Rules:
- Names: lower snake_case, unique, regex ^[a-z][a-z0-9_]*$, no subjective adjectives.
- Types: string | integer | float | boolean
    * integer = all whole numbers
    * float = any decimals / ratios
    * boolean = explicit binary
    * else use string
- No arrays, objects, composite encodings, or merged multi-concept fields.
- Descriptions: concise, objective extraction rules (no marketing/emotion/speculation).
- enum_values only for string fields with stable closed vocab; omit otherwise.
- Exclude direct outcome labels (e.g. attrition_probability, will_buy, purchase_likelihood)
    in predictive / feature engineering contexts.

Output contract:
Return exactly an InferredSchema object with JSON keys:
    - purpose (string)
    - examples_summary (string)
    - fields (array of FieldSpec objects: name, type, description, enum_values?)
    - inference_prompt (string)
""".strip()


@dataclass(frozen=True)
class SchemaInferer:
    """Infer and materialize a dynamic Pydantic model from representative examples.

    The ordered list of ``FieldSpec`` instances (``InferredSchema.fields``) is the
    single source of truth. No JSON Schema string is generated or validated.

    Attributes:
        client (OpenAI): OpenAI client used for Responses API structured parsing.
        model_name (str): Model (or Azure deployment) identifier.
    """

    client: OpenAI
    model_name: str

    def infer_schema(self, data: "SchemaInferenceInput", *args, max_retries: int = 3, **kwargs) -> "InferredSchema":
        """Infer and return an ``InferredSchema`` object.

        Args:
            data (SchemaInferenceInput): Representative example texts plus extraction purpose.
            *args: Additional positional arguments forwarded to ``client.responses.parse``.
            max_retries (int): Maximum attempts to obtain a structurally consistent schema when
                the model output fails validation. Must be >= 1.
            **kwargs: Additional keyword arguments forwarded to ``client.responses.parse``.

        Returns:
            InferredSchema: Structured schema proposal (purpose, examples summary, field specs, inference prompt).

        Raises:
            ValueError: If after retries the field list fails basic validation rules.
        """
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")

        last_err: Exception | None = None
        for attempt in range(max_retries):
            response: ParsedResponse[InferredSchema] = self.client.responses.parse(
                model=self.model_name,
                instructions=_INFER_INSTRUCTIONS,
                input=data.model_dump_json(),
                text_format=InferredSchema,
                *args,
                **kwargs,
            )
            parsed = response.output_parsed
            try:
                _basic_field_list_validation(parsed)
            except ValueError as e:
                last_err = e
                if attempt == max_retries - 1:
                    raise
                continue
            return parsed
        if last_err:  # pragma: no cover
            raise last_err
        raise RuntimeError("unreachable retry loop state")  # pragma: no cover


def _basic_field_list_validation(parsed: InferredSchema) -> None:
    names = [f.name for f in parsed.fields]
    if not names:
        raise ValueError("no fields suggested")
    if len(names) != len(set(names)):
        raise ValueError("duplicate field names detected")
    allowed = {"string", "integer", "float", "boolean"}
    for f in parsed.fields:
        if f.type not in allowed:
            raise ValueError(f"unsupported field type: {f.type}")
        if f.enum_values is not None:
            if f.type != "string":
                raise ValueError(f"enum_values only allowed for string field: {f.name}")
            if not (2 <= len(f.enum_values) <= 24):
                raise ValueError(f"enum_values length out of bounds for field {f.name}")
