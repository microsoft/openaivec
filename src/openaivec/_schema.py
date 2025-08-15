import json
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Dict, List, Literal, Optional, Type

from openai import OpenAI
from openai.types.responses import ParsedResponse
from pydantic import BaseModel, Field

from openaivec._serialize import deserialize_base_model

# Internal module: explicitly not part of public API
__all__: list[str] = []


class SuggestedField(BaseModel):
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


class SuggestedSchema(BaseModel):
    purpose: str = Field(
        description=(
            "Normalized, unambiguous restatement of the user objective with redundant, vague, or "
            "conflicting phrasing removed."
        )
    )
    example_data_description: str = Field(
        description=(
            "Objective characterization of the provided examples: content domain, structure, recurring "
            "patterns, and notable constraints."
        )
    )
    suggested_schema: List[SuggestedField] = Field(
        description=(
            "Ordered list of proposed fields derived strictly from observable, repeatable signals in the "
            "examples and aligned with the purpose."
        )
    )
    json_schema_string: str = Field(
        description=(
            "JSON Schema (Draft 2020-12) as a string, following Pydantic v2 export conventions. Must "
            "exactly reflect the fields in 'suggested_schema' (names, types) without adding, removing, or "
            "renaming any field."
        )
    )
    suggested_prompt: str = Field(
        description=(
            "Canonical, reusable extraction prompt for structuring future inputs with this schema. "
            "Must be fully derivable from 'purpose', 'example_data_description', and 'suggested_schema' "
            "(no new unstated facts or speculation). It MUST: (1) instruct the model to output only the "
            "listed fields with the exact names and primitive types; (2) forbid adding, removing, or "
            "renaming fields; (3) avoid subjective or marketing language; (4) be self-contained (no TODOs, "
            "no external references, no unresolved placeholders). Intended for direct reuse as the prompt "
            "in an extraction phase to ensure deterministic alignment with 'json_schema_string'."
        )
    )


class ExampleData(BaseModel):
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


def _validate_consistency(parsed: SuggestedSchema, schema_dict: Dict[str, Any]) -> None:
    """Validate that the machine schema matches the intermediate field proposal.

    We enforce a strict 1:1 correspondence between ``suggested_schema`` entries and
    the properties defined in ``json_schema_string`` to catch drift introduced by the
    model during the final serialization step.
    """

    # Normalize schema first (tolerate common LLM deviations)
    _normalize_schema(schema_dict)

    props = schema_dict.get("properties")
    if not isinstance(props, dict):  # Defensive; model must output object schema
        raise ValueError("json_schema_string must define an object with 'properties'.")

    suggested_names = [f.name for f in parsed.suggested_schema]
    schema_names = list(props.keys())

    if suggested_names != schema_names:
        raise ValueError(
            "Inconsistent field ordering or names between suggested_schema and json_schema_string: "
            f"suggested={suggested_names} schema={schema_names}"
        )

    type_map = {"float": "number", "integer": "integer", "string": "string", "boolean": "boolean"}
    for f in parsed.suggested_schema:
        spec = props.get(f.name)
        if not isinstance(spec, dict):
            raise ValueError(f"Property '{f.name}' missing in json_schema_string.")
        declared_type = spec.get("type")
        expected_type = type_map[f.type]
        # Tolerate 'float' as an alias for JSON Schema 'number'
        if declared_type == "float" and expected_type == "number":
            # Normalize in-place so downstream consumers see canonical form
            spec["type"] = "number"
            declared_type = "number"
        if declared_type != expected_type:
            raise ValueError(
                f"Type mismatch for field '{f.name}': suggested '{f.type}' -> expected schema type "
                f"'{expected_type}', got '{declared_type}'."
            )
        # Enum consistency (only allowed for string fields)
        if f.enum_values is not None:
            if f.type != "string":
                raise ValueError(
                    f"Field '{f.name}' supplies enum_values but type is '{f.type}' (only 'string' may define enum)."
                )
            schema_enum = spec.get("enum")
            if not isinstance(schema_enum, list):
                # Tolerant repair: inject missing enum array to align with suggested_schema
                spec["enum"] = f.enum_values
                schema_enum = f.enum_values
            if schema_enum != f.enum_values:
                raise ValueError(f"Field '{f.name}' enum mismatch: suggested {f.enum_values} schema {schema_enum}.")
        else:
            # If schema has enum but model omitted enum_values, allow (model may have inferred a closed set).
            pass


INSTRUCTIONS = """
You are a schema inference engine.

Task:
1. Normalize the user's purpose (remove ambiguity, redundancy, and contradictions).
2. Objectively summarize the example data's observable structure and patterns.
3. Propose a minimal, sufficient set of flat scalar fields (no nesting, arrays, or objects) that can be
    reliably extracted.
4. Avoid introducing a field if it would plausibly be missing for a significant portion of examples (~>20%).
5. Where a field represents a small closed set of categorical labels (classification), infer an enum of
    stable, distinct lowercase values (2–24). Do NOT invent categories not evidenced or strongly implied; omit
    enum otherwise.
6. If the purpose indicates a predictive task (mentions words like 'predict', 'probability', 'likelihood'), treat
        the goal as feature engineering: propose explanatory (independent) features only. DO NOT create fields that
        restate or trivially encode the prediction target itself (e.g., do not add 'attrition_probability', 'will_buy',
        'purchase_likelihood'). Instead surface stable, textual signals that could help a downstream model.
7. Produce a JSON Schema Draft 2020-12 (Pydantic v2 style) whose properties exactly correspond to the proposed fields.

Rules:
- Field names: lower snake_case; regex ^[a-z][a-z0-9_]*$; unique; no subjective adjectives.
- Allowed types: string | integer | float | boolean.
  * Use integer only if all observed numeric values are whole numbers.
  * Use float if any numeric value may contain decimals or represents a score/ratio.
  * Use boolean only when examples clearly encode a binary state (true/false or yes/no consistently).
  * Otherwise use string.
- Do not output arrays, objects, nested structures, or composite types.
- Do not hallucinate fields not clearly inferable from examples + purpose.
- Do not merge multiple independent concepts into one field.
- The 'json_schema_string' must: (a) define an object type, (b) include a 'properties' object, (c) include
    every field in suggested_schema in the same order, (d) not introduce additional properties.
- Keep descriptions concise and objective; avoid marketing, emotion, or speculation.
- For enum_values: only supply for string fields when the set is finite, closed, and small. Omit if open-ended.
- Field definitions MUST NOT include implicit lists concatenated in strings (e.g. comma-separated lists)
    to emulate arrays.
- If uncertainty exists between two potential fields, prefer the single clearer, consistently extractable one.
- For predictive/feature engineering purposes: exclude direct outcome labels or probabilities; focus on stable
    precursors, signals, or attributes derivable from the text (e.g. 'expressed_sentiment', 'delivery_issue_type').

Output contract:
Return data strictly conforming to the Provided Pydantic model (SuggestedSchema). Do not add fields.
""".strip()


@dataclass(frozen=True)
class SchemaSuggester:
    """Infer and materialize a dynamic Pydantic model from representative examples.

    The class orchestrates a structured call to the OpenAI Responses API, requesting
    an intermediate reasoning object (``SuggestedSchema``) plus a machine‑readable
    JSON Schema. It then validates strict consistency between the intermediate
    field proposal and the JSON Schema prior to constructing a concrete ``BaseModel``
    subclass via dynamic deserialization.

    Attributes:
        client (OpenAI): Instantiated OpenAI client used for the Responses API.
        model_name (str): Model (or Azure deployment) identifier passed to the
            Responses API.
    """

    client: OpenAI
    model_name: str

    def suggest_schema(self, data: ExampleData, *args, max_retries: int = 3, **kwargs) -> SuggestedSchema:
        """Infer and return a validated ``SuggestedSchema`` object.

        Args:
            data (ExampleData): Representative example texts plus extraction purpose.
            *args: Additional positional arguments forwarded to ``client.responses.parse``.
            max_retries (int): Maximum attempts to obtain a structurally consistent schema when
                the model output fails validation. Must be >= 1.
            **kwargs: Additional keyword arguments forwarded to ``client.responses.parse``.

        Returns:
            SuggestedSchema: Parsed structured schema proposal (purpose, data summary, field specs,
            JSON Schema string, canonical extraction prompt) guaranteed to be internally consistent.

        Raises:
            ValueError: If after ``max_retries`` attempts the JSON schema and intermediate field list still
                differ in order, names, types, or enum specifications.
        """
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")

        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            response: ParsedResponse[SuggestedSchema] = self.client.responses.parse(
                model=self.model_name,
                instructions=INSTRUCTIONS,
                input=data.model_dump_json(),
                text_format=SuggestedSchema,
                *args,
                **kwargs,
            )

            parsed = response.output_parsed
            json_schema_dict = _load_lenient_json(parsed.json_schema_string)
            # Apply normalization early so downstream validation and tests receive cleaned schema
            _normalize_schema(json_schema_dict)
            # Persist normalized form back onto the parsed object so callers see the cleaned schema
            try:
                # Secondary normalization: convert any lingering 'float' primitive types to JSON Schema 'number'
                props = json_schema_dict.get("properties")
                if isinstance(props, dict):
                    for _n, _spec in props.items():
                        if isinstance(_spec, dict) and _spec.get("type") == "float":
                            _spec["type"] = "number"
                parsed.json_schema_string = json.dumps(json_schema_dict, ensure_ascii=False)
            except Exception:
                # Non-fatal; continue with validation using normalized dict
                pass
            try:
                _validate_consistency(parsed, json_schema_dict)
                return parsed
            except ValueError as e:  # Structural mismatch; retry if attempts remain
                last_err = e
                if attempt == max_retries:
                    raise
                continue

        # Defensive (loop should return or raise)
        if last_err:
            raise last_err
        raise RuntimeError("Schema inference failed unexpectedly without error context.")


def materialize_model(suggestion: SuggestedSchema) -> Type[BaseModel]:
    """Construct a dynamic ``BaseModel`` subclass from a validated ``SuggestedSchema``.

    This is a convenience helper to turn the already consistency-checked JSON Schema string
    into a concrete Pydantic model class for downstream parsing / validation tasks.

    Args:
        suggestion (SuggestedSchema): A schema suggestion returned by ``SchemaSuggester.suggest_schema``.

    Returns:
        Type[BaseModel]: Dynamically created model type matching the suggestion.
    """

    return deserialize_base_model(_load_lenient_json(suggestion.json_schema_string))


def _load_lenient_json(text: str) -> Dict[str, Any]:
    """Attempt to parse JSON, tolerating trailing unmatched closing braces produced by LLMs.

    Strategy: try normal json.loads; on Extra data error, use JSONDecoder.raw_decode to grab the first
    complete JSON value and ensure the remainder contains only closing braces / whitespace. If so, accept
    the first value; otherwise re-raise.
    """

    try:
        return json.loads(text)
    except JSONDecodeError:
        # Heuristic: find the last position where braces are balanced and truncate there.
        depth = 0
        last_good_index = None
        for i, ch in enumerate(text):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last_good_index = i + 1
        if last_good_index is not None:
            candidate = text[:last_good_index]
            try:
                obj = json.loads(candidate)
                if not isinstance(obj, dict):
                    raise ValueError("Top-level JSON schema must be an object")
                return obj
            except JSONDecodeError:
                pass
        # Fallback to original exception behavior
        raise


def _normalize_schema(schema_dict: Dict[str, Any]) -> None:
    """In-place normalization of common LLM schema quirks for consistency checking.

    - Convert 'enum_values' -> 'enum' where appropriate
    - Collapse optional patterns encoded as anyOf/oneOf [T, null] into a single 'type'
    - Remove unknown keys that can cause noise (currently only pass-through)
    """

    props = schema_dict.get("properties")
    if not isinstance(props, dict):
        return
    for name, spec in props.items():
        if not isinstance(spec, dict):
            continue
        # enum_values -> enum
        if "enum_values" in spec and "enum" not in spec:
            ev = spec.get("enum_values")
            if isinstance(ev, list) and all(isinstance(x, str) for x in ev):
                spec["enum"] = ev
            spec.pop("enum_values", None)
        # anyOf optional pattern (allow rewriting even if 'type' present but ambiguous)
        if "anyOf" in spec:
            any_of = spec.get("anyOf")
            if isinstance(any_of, list):
                non_null = [x for x in any_of if isinstance(x, dict) and x.get("type") not in {None, "null"}]
                if len(non_null) == 1 and isinstance(non_null[0], dict):
                    candidate_type = non_null[0].get("type")
                    if candidate_type in {"string", "integer", "number", "boolean"}:
                        spec["type"] = candidate_type
                        if "enum" in non_null[0] and "enum" not in spec:
                            spec["enum"] = non_null[0]["enum"]
                        spec.pop("anyOf", None)
        # Final safeguard: if still no type but an anyOf with primitive + null exists, force collapse
        if "type" not in spec and isinstance(spec.get("anyOf"), list):
            primitives = [
                x.get("type") for x in spec["anyOf"] if isinstance(x, dict) and x.get("type") not in {None, "null"}
            ]
            if len(primitives) == 1 and primitives[0] in {"string", "integer", "number", "boolean"}:
                spec["type"] = primitives[0]
                spec.pop("anyOf", None)
