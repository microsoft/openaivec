"""Internal schema inference & dynamic model materialization utilities.

This (non-public) module converts a small *representative* sample of free‑text
examples plus an *instructions* statement into:

1. A vetted hierarchical object specification (``ObjectSpec``) whose recursively
     defined ``fields`` (``FieldSpec``) capture reliably extractable signals.
2. A reusable, self‑contained extraction prompt (``inference_prompt``) that
     freezes the agreed schema contract (no additions / renames / omissions).
3. A dynamically generated Pydantic model mirroring the hierarchical schema,
     enabling immediate typed parsing with the OpenAI Responses API.
4. A ``PreparedTask`` wrapper (``InferredSchema.task``) for downstream batched
     responses / structured extraction flows in pandas or Spark.

Core goals:
* Minimize manual, subjective schema design iterations.
* Enforce objective naming / typing / enum rules early (guard rails rather than
    after‑the‑fact cleaning).
* Provide deterministic reusability: the same prompt + model yield stable field
    ordering & types for analytics or feature engineering.
* Avoid outcome / target label leakage in predictive (feature engineering)
    contexts by explicitly excluding direct target restatements.

This module is intentionally **internal** (``__all__ = []``). Public users
should interact through higher‑level batch APIs once a schema has been inferred.

Design constraints (updated):
* Root: single ``ObjectSpec`` (UpperCamelCase name) containing one or more fields.
* Field types: string | integer | float | boolean | enum | object |
    string_array | integer_array | float_array | boolean_array | enum_array | object_array
* Arrays are homogeneous lists of their base type.
* Nested objects / arrays of objects are allowed when semantically cohesive; keep
    depth shallow and avoid gratuitous nesting.
* Enumerations use ``enum_spec`` with explicit ``name`` (UpperCamelCase) and 1–24
    raw label values (project constant). Values collapse by uppercasing; order not guaranteed.
* Field names: lower_snake_case; unique per containing object.
* Boolean names: affirmative 'is_' prefix.
* Numeric (integer/float) names encode unit / measure suffix (e.g. *_count, *_ratio, *_ms).
* Validation retries ensure a structurally coherent suggestion before returning.

Example (conceptual):
        from openai import OpenAI
        client = OpenAI()
        inferer = SchemaInferer(client=client, model_name="gpt-4.1-mini")
        schema = inferer.infer_schema(
                SchemaInferenceInput(
                        examples=["Order #123 delayed due to weather", "Order #456 delivered"],
                        instructions="Extract operational status signals for logistics analytics",
                )
        )
        Model = schema.model  # dynamic Pydantic model
        task = schema.task    # PreparedTask for batch extraction

The implementation purposefully does *not* emit or depend on JSON Schema; the
authoritative contract is the recursive ``ObjectSpec`` tree.
"""

from dataclasses import dataclass

from openai import OpenAI
from openai.types.responses import ParsedResponse
from pydantic import BaseModel, Field

from openaivec._dynamic import ObjectSpec, _build_model
from openaivec._model import PreparedTask

# Internal module: explicitly not part of public API
__all__: list[str] = []


class InferredSchema(BaseModel):
    """Result of a schema inference round.

    Contains the normalized *instructions*, objective *examples_summary*, the root
    hierarchical ``object_spec`` contract, and the canonical reusable
    ``inference_prompt``. The prompt MUST be fully derivable from the other
    components (no new unstated facts) to preserve traceability.

    Attributes:
        instructions: Unambiguous restatement of the user's objective.
        examples_summary: Neutral description of structural / semantic patterns
            observed in the examples.
        examples_instructions_alignment: Mapping from instructions facets to concrete
            recurring evidence (or explicit gaps) anchoring extraction scope.
        object_spec: Root ``ObjectSpec`` (UpperCamelCase name) whose ``fields``
            recursively define the extraction schema.
        inference_prompt: Canonical instructions enforcing exact field names,
            hierarchy, and types (no additions/removals/renames).
    """

    instructions: str = Field(
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
    examples_instructions_alignment: str = Field(
        description=(
            "Explanation of how observable recurring patterns in the examples substantiate and bound the stated "
            "instructions. Should reference instructions facets and cite supporting example evidence (or note any "
            "gaps) to reduce hallucinated fields. Internal diagnostic / quality aid; not required for downstream "
            "extraction."
        )
    )
    object_spec: ObjectSpec = Field(
        description=(
            "Root ObjectSpec (recursive). Each contained object's field list is unique-name ordered and derived "
            "strictly from observable, repeatable signals aligned with the instructions."
        )
    )
    inference_prompt: str = Field(
        description=(
            "Canonical, reusable extraction prompt. Must be derivable from instructions + summaries + object_spec. "
            "Enforces exact hierarchical field set (names, order per object, types) forbidding additions, removals, "
            "renames, or subjective language. Self-contained (no TODOs, external refs, or placeholders)."
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
    def model(self) -> type[BaseModel]:
        """Dynamically materialized Pydantic model for the inferred schema.

        Equivalent to calling :meth:`build_model` each access (not cached).

        Returns:
            type[BaseModel]: Fresh model type reflecting ``fields`` ordering.
        """
        return self.build_model()

    @property
    def task(self) -> PreparedTask:
        """PreparedTask integrating the schema's extraction prompt & model.

        Returns:
            PreparedTask: Ready for batched structured extraction calls.
        """
        return PreparedTask(
            instructions=self.inference_prompt,
            response_format=self.model,
            api_kwargs={"top_p": None, "temperature": None},
        )

    def build_model(self) -> type[BaseModel]:
        return _build_model(self.object_spec)

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
        instructions: Plain language description of downstream usage (analytics,
            filtering, enrichment, feature engineering, etc.). Guides field
            relevance & exclusion of outcome labels.
    """

    examples: list[str] = Field(
        description=(
            "Representative sample texts (strings). Provide only data the schema should generalize over; "
            "exclude outliers not in scope."
        )
    )
    instructions: str = Field(
        description=(
            "Plain language statement describing the downstream use of the extracted structured data (e.g. "
            "analytics, filtering, enrichment)."
        )
    )


_INFER_INSTRUCTIONS = """
You are a schema inference engine.

Task:
1. Normalize the user's instructions (eliminate ambiguity, redundancy, contradictions).
2. Objectively summarize observable patterns in the example texts.
3. Produce an "examples_instructions_alignment" explanation mapping instructions facets to concrete recurring
     evidence (or gaps).
4. Propose a minimal hierarchical schema (root ObjectSpec) comprised of reliably extractable fields. Use nesting ONLY
     when a group of fields forms a cohesive sub-entity repeated in the data; otherwise keep flat.
5. Skip fields likely missing in a large share (>~20%) of realistic inputs.
6. Provide enum_spec ONLY when a small stable closed categorical set (1–{_MAX_ENUM_VALUES} raw tokens) is clearly
     evidenced; never invent unseen categories.
7. If the instructions indicate prediction (predict / probability / likelihood),
   output only explanatory features (no target restatement).

Rules:
- Field names: lower snake_case, unique within each object, regex ^[a-z][a-z0-9_]*$, no subjective adjectives.
- Field types: string | integer | float | boolean | enum | object | string_array | integer_array | float_array |
    boolean_array | enum_array | object_array
    * *_array are homogeneous lists of their primitive / enum / object base type.
    * Use object/object_array ONLY for semantically cohesive grouped attributes; avoid gratuitous layers.
- Enumerations: use enum_spec { name (UpperCamelCase), values [raw_tokens...] }. values length 1–{_MAX_ENUM_VALUES}.
    Use ONLY when closed set is evidenced. Otherwise, use string.
- Numeric (integer|float) names encode explicit unit/measure suffix (e.g. *_count, *_seconds, *_usd, *_ratio, *_score).
- Boolean names start with 'is_' followed by positive predicate (no negations like is_not_*).
- Array field names SHOULD end with '_array' for primitive/enum arrays; object_array
    fields may use plural noun or *_array pattern.
- Descriptions: concise, objective extraction criteria (no marketing/emotion/speculation).
- Exclude direct outcome labels in predictive contexts.
- Avoid superficial renames; semantic transformation only.
- Keep total field count focused (typically <= 16) optimizing for reusable analytical / ML features.

Output contract:
Return exactly an InferredSchema JSON object with keys:
        - instructions (string)
        - examples_summary (string)
        - examples_instructions_alignment (string)
        - object_spec (ObjectSpec: name, fields[list[FieldSpec]])
        - inference_prompt (string)
Where each FieldSpec includes: name, type, description, optional enum_spec (for
enum / enum_array), optional object_spec (for object / object_array).
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

    def infer_schema(self, data: SchemaInferenceInput, *args, max_retries: int = 8, **kwargs) -> InferredSchema:
        """Infer a validated schema from representative examples.

          Workflow:
                1. Submit ``SchemaInferenceInput`` (JSON) + instructions via
                    ``responses.parse`` requesting an ``InferredSchema`` object.
                2. Attempt dynamic model build (``parsed.build_model()``) which performs recursive
                    structural validation (names, types, enum/object specs) via the dynamic layer.
                3. Retry (up to ``max_retries``) on validation failure.

        Args:
            data (SchemaInferenceInput): Representative examples + instructions.
            *args: Positional passthrough to ``client.responses.parse``.
            max_retries (int, optional): Attempts before surfacing the last validation error
                (must be >= 1). Defaults to 3.
            **kwargs: Keyword passthrough to ``client.responses.parse``.

        Returns:
            InferredSchema: Fully validated schema (instructions, examples summary,
            ordered fields, extraction prompt).

        Raises:
            ValueError: Validation still fails after exhausting retries.
        """
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")

        last_err: Exception | None = None
        previous_errors: list[str] = []
        for attempt in range(max_retries):
            if attempt == 0:
                instructions = _INFER_INSTRUCTIONS
            else:
                # Provide structured feedback for correction. Keep concise and prohibit speculative expansion.
                feedback_lines = [
                    "--- PRIOR VALIDATION FEEDBACK ---",
                ]
                for i, err in enumerate(previous_errors[-5:], 1):  # include last up to 5 errors
                    feedback_lines.append(f"{i}. {err}")
                feedback_lines.extend(
                    [
                        "Adjust ONLY listed issues; avoid adding brand-new fields unless essential.",
                        "Don't hallucinate or broaden enum_values unless enum rule caused failure.",
                        "Duplicate names: minimally rename; keep semantics.",
                        "Unsupported type: change to string|integer|float|boolean (no new facts).",
                        "Bad enum length: drop enum or constrain to 2–24 evidenced tokens.",
                    ]
                )
                instructions = _INFER_INSTRUCTIONS + "\n\n" + "\n".join(feedback_lines)

            response: ParsedResponse[InferredSchema] = self.client.responses.parse(
                model=self.model_name,
                instructions=instructions,
                input=data.model_dump_json(),
                text_format=InferredSchema,
                *args,
                **kwargs,
            )
            parsed = response.output_parsed
            try:
                # Validate the field list structure
                parsed.build_model()
                return parsed
            except ValueError as e:
                last_err = e
                previous_errors.append(str(e))
                if attempt == max_retries - 1:
                    raise ValueError(
                        f"Schema validation failed after {max_retries} attempts. Last error: {last_err}"
                    ) from last_err

        if last_err:
            raise last_err
        raise RuntimeError("unreachable retry loop state")
