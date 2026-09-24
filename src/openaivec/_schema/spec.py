from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, create_model

__all__: list[str] = []

_MAX_ENUM_VALUES = 24
_MIN_ENUM_VALUES = 1


class FieldSpec(BaseModel):
    name: str = Field(
        description=(
            "Field name in lower_snake_case. Rules: (1) Use only lowercase letters, numbers, and underscores; "
            "must start with a letter. (2) For numeric quantities append an explicit unit (e.g. 'duration_seconds', "
            "'price_usd'). (3) Boolean fields use an affirmative 'is_' prefix (e.g. 'is_active'); avoid negative / "
            "ambiguous forms like 'is_deleted' (prefer 'is_active', 'is_enabled'). (4) Name must be unique within the "
            "containing object."
        )
    )
    type: Literal[
        "string",
        "integer",
        "float",
        "boolean",
        "enum",
        "object",
        "string_array",
        "integer_array",
        "float_array",
        "boolean_array",
        "enum_array",
        "object_array",
    ] = Field(
        description=(
            "Logical data type. Allowed values: string | integer | float | boolean | enum | object | string_array | "
            "integer_array | float_array | boolean_array | enum_array | object_array. *_array variants represent a "
            "homogeneous list of the base type. 'enum' / 'enum_array' require 'enum_spec'. 'object' / 'object_array' "
            "require 'object_spec'. Primitives must not define 'enum_spec' or 'object_spec'."
        )
    )
    description: str = Field(
        description=(
            "Human‑readable, concise explanation of the field's meaning and business intent. Should clarify units, "
            "value semantics, and any domain constraints not captured by type. 1–2 sentences; no implementation notes."
        )
    )
    enum_spec: EnumSpec | None = Field(
        default=None,
        description=(
            "Enumeration specification for 'enum' / 'enum_array'. Must be provided (non-empty) for those types and "
            "omitted for all others. Maximum size enforced by constant."
        ),
    )
    object_spec: ObjectSpec | None = Field(
        default=None,
        description=(
            "Nested object schema. Required for 'object' / 'object_array'; must be omitted for every other type. The "
            "contained 'name' is used to derive the generated nested Pydantic model class name."
        ),
    )
    minimum: float | None = Field(default=None, description="Inclusive lower bound for an integer or float field.")
    maximum: float | None = Field(default=None, description="Inclusive upper bound for an integer or float field.")
    boolean_value: bool | None = Field(default=None, description="Required value for a boolean field, if fixed.")
    nullable: bool = Field(default=False, description="Whether the required field may contain null.")


class EnumSpec(BaseModel):
    """Enumeration specification for enum / enum_array field types.

    Attributes:
        name: Required Enum class name (UpperCamelCase). Must match ^[A-Z][A-Za-z0-9]*$. Previously optional; now
            explicit to remove implicit coupling to the field name and make schemas self‑describing.
        values: Exact string labels (1–_MAX_ENUM_VALUES before de-dup). Exact duplicates
            are removed in order; casing variants remain distinct.
    """

    name: str = Field(
        description=("Required Enum class name (UpperCamelCase). Valid pattern: ^[A-Z][A-Za-z0-9]*$."),
    )
    values: list[str] = Field(
        description=(
            f"Exact enum string values ({_MIN_ENUM_VALUES}–{_MAX_ENUM_VALUES}). "
            "Duplicate values are removed while preserving first-seen order."
        )
    )


class ObjectSpec(BaseModel):
    name: str = Field(
        description=(
            "Object model class name in UpperCamelCase (singular noun). Must match ^[A-Z][A-Za-z0-9]*$ and is used "
            "directly as the generated Pydantic model class name (no transformation)."
        )
    )
    fields: list[FieldSpec] = Field(
        description=(
            "Non-empty list of FieldSpec definitions composing the object. Each field name must be unique; order is "
            "preserved in the generated model."
        )
    )


def _string_enum(enum_spec: EnumSpec) -> type[Enum]:
    members: dict[str, str] = {}
    for index, value in enumerate(dict.fromkeys(enum_spec.values), 1):
        if not value or not value.strip():
            raise ValueError("enum_spec.values must contain non-blank strings.")
        candidate = value.upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", candidate) or candidate.startswith("VALUE_"):
            candidate = f"VALUE_{index}"
        name = candidate
        suffix = 2
        while name in members:
            name = f"{candidate}_{suffix}"
            suffix += 1
        members[name] = value
    return cast(type[Enum], Enum(enum_spec.name, members, type=str))


def _field_constraints(field: FieldSpec) -> dict[str, Any]:
    numeric = field.type in {"integer", "float"}
    if (field.minimum is not None or field.maximum is not None) and not numeric:
        raise ValueError(f"Field '{field.name}': minimum/maximum require an integer or float field.")
    if field.minimum is not None and field.maximum is not None and field.minimum > field.maximum:
        raise ValueError(f"Field '{field.name}': minimum must not exceed maximum.")
    if field.type == "integer":
        for bound in (field.minimum, field.maximum):
            if bound is not None and (not float("-inf") < bound < float("inf") or not bound.is_integer()):
                raise ValueError(f"Field '{field.name}': integer bounds must be finite whole numbers.")
    if numeric:
        for bound in (field.minimum, field.maximum):
            if bound is not None and not float("-inf") < bound < float("inf"):
                raise ValueError(f"Field '{field.name}': numeric bounds must be finite.")
    if field.boolean_value is not None and field.type != "boolean":
        raise ValueError(f"Field '{field.name}': boolean_value requires a boolean field.")
    return {"ge": field.minimum, "le": field.maximum}


def _build_model(model_spec: ObjectSpec) -> type[BaseModel]:
    lower_sname_pattern = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
    upper_camel_pattern = re.compile(r"^[A-Z][A-Za-z0-9]*$")
    type_map: dict[str, type] = {
        "string": str,
        "integer": int,
        "float": float,
        "boolean": bool,
        "string_array": list[str],
        "integer_array": list[int],
        "float_array": list[float],
        "boolean_array": list[bool],
    }
    output_fields: dict[str, tuple[Any, object]] = {}

    if not upper_camel_pattern.fullmatch(model_spec.name):
        raise ValueError(f"Object name '{model_spec.name}' must be UpperCamelCase.")
    if not model_spec.fields:
        raise ValueError(f"Object '{model_spec.name}' must contain at least one field.")
    field_names: list[str] = [field.name for field in model_spec.fields]

    # Assert that names of fields are not duplicated
    if len(field_names) != len(set(field_names)):
        raise ValueError("Field names must be unique within the object spec.")

    for field in model_spec.fields:
        constraints = _field_constraints(field)
        # Assert that field names are lower_snake_case
        if not lower_sname_pattern.match(field.name):
            raise ValueError(f"Field name '{field.name}' must be in lower_snake_case format (e.g., 'my_field_name').")

        # (EnumSpec.name now mandatory; no need to derive a fallback name from the field.)
        match field:
            case FieldSpec(
                name=name,
                type="string"
                | "integer"
                | "float"
                | "boolean"
                | "string_array"
                | "integer_array"
                | "float_array"
                | "boolean_array",
                description=description,
                enum_spec=None,
                object_spec=None,
            ):
                field_type = type_map[field.type]
                if field.type == "boolean" and field.boolean_value is not None:
                    field_type = Literal[True] if field.boolean_value else Literal[False]
                output_fields[name] = (field_type, Field(description=description, **constraints))

            case FieldSpec(name=name, type="enum", description=description, enum_spec=enum_spec, object_spec=None) if (
                enum_spec
                and _MIN_ENUM_VALUES <= len(enum_spec.values) <= _MAX_ENUM_VALUES
                and upper_camel_pattern.match(enum_spec.name)
            ):
                enum_type = _string_enum(enum_spec)
                output_fields[name] = (enum_type, Field(description=description))

            case FieldSpec(
                name=name, type="enum_array", description=description, enum_spec=enum_spec, object_spec=None
            ) if (
                enum_spec
                and _MIN_ENUM_VALUES <= len(enum_spec.values) <= _MAX_ENUM_VALUES
                and upper_camel_pattern.match(enum_spec.name)
            ):
                enum_type = _string_enum(enum_spec)
                output_fields[name] = (list[enum_type], Field(description=description))

            case FieldSpec(
                name=name, type="object", description=description, enum_spec=None, object_spec=nested_object_spec
            ) if nested_object_spec and upper_camel_pattern.match(nested_object_spec.name):
                nested_model = _build_model(nested_object_spec)
                output_fields[name] = (nested_model, Field(description=description))

            case FieldSpec(
                name=name, type="object_array", description=description, enum_spec=None, object_spec=nested_object_spec
            ) if nested_object_spec and upper_camel_pattern.match(nested_object_spec.name):
                nested_model = _build_model(nested_object_spec)
                output_fields[name] = (list[nested_model], Field(description=description))

            # ---- Error cases (explicit reasons) ----
            # Enum type without enum_spec (None or empty)
            case FieldSpec(
                name=name,
                type="enum",
                enum_spec=enum_spec,
                object_spec=None,
            ) if not enum_spec or not enum_spec.values:
                raise ValueError(f"Field '{name}': enum type requires non-empty enum_spec values list.")
            # Enum type exceeding max length
            case FieldSpec(
                name=name,
                type="enum",
                enum_spec=enum_spec,
                object_spec=None,
            ) if enum_spec and len(enum_spec.values) > _MAX_ENUM_VALUES:
                raise ValueError(
                    (
                        f"Field '{name}': enum type supports at most {_MAX_ENUM_VALUES} enum_spec values "
                        f"(got {len(enum_spec.values)})."
                    )
                )
            # Enum type invalid explicit name pattern
            case FieldSpec(
                name=name,
                type="enum",
                enum_spec=enum_spec,
                object_spec=None,
            ) if enum_spec and not upper_camel_pattern.match(enum_spec.name):
                raise ValueError(
                    (f"Field '{name}': enum_spec.name '{enum_spec.name}' invalid – must match ^[A-Z][A-Za-z0-9]*$")
                )
            # Enum type incorrectly provides an object_spec
            case FieldSpec(
                name=name,
                type="enum",
                enum_spec=enum_spec,
                object_spec=field_object_spec,
            ) if field_object_spec is not None:
                raise ValueError(
                    f"Field '{name}': enum type must not provide object_spec (got object_spec={field_object_spec!r})."
                )
            # Enum array type without enum_spec
            case FieldSpec(
                name=name,
                type="enum_array",
                enum_spec=enum_spec,
                object_spec=None,
            ) if not enum_spec or not enum_spec.values:
                raise ValueError(f"Field '{name}': enum_array type requires non-empty enum_spec values list.")
            # Enum array type exceeding max length
            case FieldSpec(
                name=name,
                type="enum_array",
                enum_spec=enum_spec,
                object_spec=None,
            ) if enum_spec and len(enum_spec.values) > _MAX_ENUM_VALUES:
                raise ValueError(
                    (
                        f"Field '{name}': enum_array type supports at most {_MAX_ENUM_VALUES} enum_spec values "
                        f"(got {len(enum_spec.values)})."
                    )
                )
            # Enum array type invalid explicit name pattern
            case FieldSpec(
                name=name,
                type="enum_array",
                enum_spec=enum_spec,
                object_spec=None,
            ) if enum_spec and not upper_camel_pattern.match(enum_spec.name):
                raise ValueError(
                    (f"Field '{name}': enum_spec.name '{enum_spec.name}' invalid – must match ^[A-Z][A-Za-z0-9]*$")
                )
            # Enum array type incorrectly provides an object_spec
            case FieldSpec(
                name=name,
                type="enum_array",
                enum_spec=enum_spec,
                object_spec=field_object_spec,
            ) if field_object_spec is not None:
                raise ValueError(
                    (
                        f"Field '{name}': enum_array type must not provide object_spec "
                        f"(got object_spec={field_object_spec!r})."
                    )
                )
            # Object type missing object_spec
            case FieldSpec(
                name=name,
                type="object",
                enum_spec=enum_spec,
                object_spec=None,
            ):
                raise ValueError(f"Field '{name}': object type requires object_spec (got object_spec=None).")
            # Object array type missing object_spec
            case FieldSpec(
                name=name,
                type="object_array",
                enum_spec=enum_spec,
                object_spec=None,
            ):
                raise ValueError(f"Field '{name}': object_array type requires object_spec (got object_spec=None).")
            # Object/object_array provided but invalid name pattern
            case FieldSpec(
                name=name,
                type="object" | "object_array",
                enum_spec=enum_spec,
                object_spec=field_object_spec,
            ) if field_object_spec is not None and not upper_camel_pattern.match(field_object_spec.name):
                raise ValueError(
                    (
                        f"Field '{name}': object_spec.name '{field_object_spec.name}' must be UpperCamelCase "
                        "(regex ^[A-Z][A-Za-z0-9]*$) and contain only letters and digits."
                    )
                )
            # Object/object_array types must not provide enum_spec
            case FieldSpec(
                name=name,
                type="object" | "object_array",
                enum_spec=enum_spec,
                object_spec=field_object_spec,
            ) if enum_spec is not None:
                raise ValueError(
                    f"Field '{name}': {field.type} must not define enum_spec (got enum_spec={enum_spec!r})."
                )
            # Primitive / simple array types must not have enum_spec
            case FieldSpec(
                name=name,
                type="string"
                | "integer"
                | "float"
                | "boolean"
                | "string_array"
                | "integer_array"
                | "float_array"
                | "boolean_array",
                enum_spec=enum_spec,
                object_spec=field_object_spec,
            ) if enum_spec is not None:
                raise ValueError(
                    (f"Field '{name}': type '{field.type}' must not define enum_spec (got enum_spec={enum_spec!r}).")
                )
            # Primitive / simple array types must not have object_spec
            case FieldSpec(
                name=name,
                type="string"
                | "integer"
                | "float"
                | "boolean"
                | "string_array"
                | "integer_array"
                | "float_array"
                | "boolean_array",
                enum_spec=None,
                object_spec=field_object_spec,
            ) if field_object_spec is not None:
                raise ValueError(
                    (
                        f"Field '{name}': type '{field.type}' must not define object_spec "
                        f"(got object_spec={field_object_spec!r})."
                    )
                )
            # Any other unmatched combination
            case FieldSpec() as f:
                raise ValueError(
                    (
                        "Field configuration invalid / unrecognized combination: "
                        f"name={f.name!r}, type={f.type!r}, enum_spec={'set' if f.enum_spec else None}, "
                        f"object_spec={'set' if f.object_spec else None}."
                    )
                )

        if field.nullable:
            annotation, metadata = output_fields[field.name]
            output_fields[field.name] = (annotation | None, metadata)

    return create_model(model_spec.name, __config__=ConfigDict(extra="forbid"), **cast(Any, output_fields))
