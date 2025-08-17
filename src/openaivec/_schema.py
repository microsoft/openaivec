"""Internal schema inference & dynamic model materialization utilities.

This (non-public) module converts a small *representative* sample of free‑text
examples plus a *purpose* statement into:

1. A vetted, flat list of scalar field specifications (``FieldSpec``) that can
    be *reliably* extracted across similar future inputs.
2. A reusable, self‑contained extraction prompt (``inference_prompt``) that
    freezes the agreed schema contract (no additions / renames / omissions).
3. A dynamically generated Pydantic model whose fields mirror the inferred
    schema, enabling immediate typed parsing with the OpenAI Responses API.
4. A ``PreparedTask`` wrapper (``InferredSchema.task``) for downstream batched
    responses/structured extraction flows in pandas or Spark.

Core goals:
* Minimize manual, subjective schema design iterations.
* Enforce objective naming / typing / enum rules early (guard rails rather than
  after‑the‑fact cleaning).
* Provide deterministic reusability: the same prompt + model yield stable
  column ordering & types for analytics or feature engineering.
* Avoid outcome / target label leakage in predictive (feature engineering)
  contexts by explicitly excluding direct target restatements.

This module is intentionally **internal** (``__all__ = []``). Public users
should interact through higher‑level batch APIs once a schema has been inferred.

Design constraints:
* Flat schema only (no nesting / arrays) to simplify Spark & pandas alignment.
* Primitive types limited to {string, integer, float, boolean}.
* Optional enumerations for *closed*, *observed* categorical sets only.
* Validation retries ensure a structurally coherent suggestion before returning.

Example (conceptual):
     from openai import OpenAI
     client = OpenAI()
     inferer = SchemaInferer(client=client, model_name="gpt-4.1-mini")
     schema = inferer.infer_schema(
          SchemaInferenceInput(
                examples=["Order #123 delayed due to weather", "Order #456 delivered"],
                purpose="Extract operational status signals for logistics analytics",
          )
     )
     Model = schema.model  # dynamic Pydantic model
     task = schema.task    # PreparedTask for batch extraction

The implementation purposefully does *not* emit or depend on JSON Schema; the
authoritative contract is the ordered list of ``FieldSpec`` instances.
"""

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
    """Specification for a single candidate output field.

    Each ``FieldSpec`` encodes a *flat*, scalar, semantically atomic unit the
    model should extract. These become columns in downstream DataFrames.

    Validation focuses on: objective naming, primitive typing, and *optional*
    closed categorical vocabularies. Enumerations are intentionally conservative
    (must derive from clear evidence) to reduce over‑fitted schemas.

    Attributes:
        name: Lower snake_case unique identifier (regex ^[a-z][a-z0-9_]*$). Avoid
            subjective modifiers ("best", "great", "high_quality").
        type: One of ``string|integer|float|boolean``. ``integer`` only if all
            observed numeric values are whole numbers; ``float`` if any decimal
            or ratio appears. ``boolean`` strictly for explicit binary forms.
        description: Concise, objective extraction rule (what qualifies / what
            to ignore). Disambiguate from overlapping fields if needed.
        enum_values: Optional stable closed set of lowercase string labels
            (2–24). Only for *string* type when the vocabulary is clearly
            evidenced; never hallucinate or extrapolate.
    """

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
    """Result of a schema inference round.

    Contains the normalized *purpose*, an objective *examples_summary*, the
    ordered ``fields`` contract, and the canonical reusable ``inference_prompt``.

    The prompt is constrained to be fully derivable from the other components;
    adding novel unstated facts is disallowed to preserve traceability.

    Attributes:
        purpose: Unambiguous restatement of the user's objective (noise &
            redundancy removed).
        examples_summary: Neutral description of structural / semantic patterns
            observed in the examples (domain, recurring signals, constraints).
        fields: Ordered list of ``FieldSpec`` objects comprising the schema's
            sole authoritative contract.
        inference_prompt: Self-contained extraction instructions enforcing an
            exact field set (names, order, primitive types) with prohibition on
            alterations or subjective flourishes.
    """

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
            path (str): Path to a UTF‑8 JSON document previously produced via ``save``.

        Returns:
            InferredSchema: Reconstructed instance.
        """
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())

    @property
    def model(self) -> Type[BaseModel]:
        """Dynamically materialized Pydantic model for the inferred schema.

        Equivalent to calling :meth:`build_model` each access (not cached).

        Returns:
            Type[BaseModel]: Fresh model type reflecting ``fields`` ordering.
        """
        return self.build_model()

    @property
    def task(self) -> PreparedTask:
        """PreparedTask integrating the schema's extraction prompt & model.

        Returns:
            PreparedTask: Ready for batched structured extraction calls.
        """
        return PreparedTask(
            instructions=self.inference_prompt, response_format=self.model, top_p=None, temperature=None
        )

    def build_model(self) -> Type[BaseModel]:
        """Create a new dynamic ``BaseModel`` class adhering to this schema.

        Implementation details:
            * Maps primitive types: string→``str``, integer→``int``, float→``float``, boolean→``bool``.
            * For enumerated string fields, constructs an ad‑hoc ``Enum`` subclass with
              stable member names (collision‑safe, normalized to ``UPPER_SNAKE``).
            * All fields are required (ellipsis ``...``). Optionality can be
              introduced later by modifying this logic if needed.

        Returns:
            Type[BaseModel]: New (not cached) model type; order matches ``fields``.
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
        """Persist this inferred schema as pretty‑printed JSON.

        Args:
            path (str): Destination filesystem path.
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))


class SchemaInferenceInput(BaseModel):
    """Input payload for schema inference.

    Attributes:
        examples: Representative sample texts restricted to the in‑scope
            distribution (exclude outliers / noise). Size should be *minimal*
            yet sufficient to surface recurring patterns.
        purpose: Plain language description of downstream usage (analytics,
            filtering, enrichment, feature engineering, etc.). Guides field
            relevance & exclusion of outcome labels.
    """

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
    """High-level orchestrator for schema inference against the Responses API.

    Responsibilities:
        * Issue a structured parsing request with strict instructions.
        * Retry (up to ``max_retries``) when the produced field list violates
          baseline structural rules (duplicate names, unsupported types, etc.).
        * Return a fully validated ``InferredSchema`` ready for dynamic model
          generation & downstream batch extraction.

    The inferred schema intentionally avoids JSON Schema intermediates; the
    authoritative contract is the ordered ``FieldSpec`` list.

    Attributes:
        client: OpenAI client for calling ``responses.parse``.
        model_name: Model / deployment identifier.
    """

    client: OpenAI
    model_name: str

    def infer_schema(self, data: "SchemaInferenceInput", *args, max_retries: int = 3, **kwargs) -> "InferredSchema":
        """Infer a validated schema from representative examples.

        Workflow:
            1. Submit ``SchemaInferenceInput`` (JSON) + instructions via
               ``responses.parse`` requesting an ``InferredSchema`` object.
            2. Validate the returned field list with ``_basic_field_list_validation``.
            3. Retry (up to ``max_retries``) if validation fails.

        Args:
            data (SchemaInferenceInput): Representative examples + purpose.
            *args: Positional passthrough to ``client.responses.parse``.
            max_retries (int, optional): Attempts before surfacing the last validation error
                (must be >= 1). Defaults to 3.
            **kwargs: Keyword passthrough to ``client.responses.parse``.

        Returns:
            InferredSchema: Fully validated schema (purpose, examples summary,
            ordered fields, extraction prompt).

        Raises:
            ValueError: Validation still fails after exhausting retries.
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
    """Lightweight structural validation of an inferred field list.

    Checks:
        * Non-empty field set.
        * No duplicate names.
        * All types in the allowed primitive set.
        * ``enum_values`` only on string fields and size within bounds (2–24).

    Args:
        parsed (InferredSchema): Candidate ``InferredSchema`` instance.

    Raises:
        ValueError: Any invariant is violated.
    """
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
