from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, create_model

__all__: list[str] = []

_MAX_ENUM_VALUES = 24


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


class EnumSpec(BaseModel):
    """Enumeration specification for enum / enum_array field types.

    Attributes:
        name: Required Enum class name (UpperCamelCase). Must match ^[A-Z][A-Za-z0-9]*$. Previously optional; now
            explicit to remove implicit coupling to the field name and make schemas self‑describing.
        values: Raw label values (1–_MAX_ENUM_VALUES before de‑dup). Values are uppercased then
            de-duplicated using a set; ordering of generated Enum members is not guaranteed. Any
            casing variants collapse silently to a single member.
    """

    name: str = Field(
        description=("Required Enum class name (UpperCamelCase). Valid pattern: ^[A-Z][A-Za-z0-9]*$."),
    )
    values: list[str] = Field(
        description=(
            f"Raw enum label values (1–{_MAX_ENUM_VALUES}). Uppercased then deduplicated; order of members "
            "not guaranteed."
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


def _build_model(object_spec: ObjectSpec) -> type[BaseModel]:
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
    output_fields: dict[str, tuple[type, object]] = {}

    field_names: list[str] = [field.name for field in object_spec.fields]

    # Assert that names of fields are not duplicated
    if len(field_names) != len(set(field_names)):
        raise ValueError("Field names must be unique within the object spec.")

    for field in object_spec.fields:
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
                output_fields[name] = (field_type, Field(description=description))

            case FieldSpec(name=name, type="enum", description=description, enum_spec=enum_spec, object_spec=None) if (
                enum_spec
                and 0 < len(enum_spec.values) <= _MAX_ENUM_VALUES
                and upper_camel_pattern.match(enum_spec.name)
            ):
                member_names = list({v.upper() for v in enum_spec.values})
                enum_type = Enum(enum_spec.name, member_names)
                output_fields[name] = (enum_type, Field(description=description))

            case FieldSpec(
                name=name, type="enum_array", description=description, enum_spec=enum_spec, object_spec=None
            ) if (
                enum_spec
                and 0 < len(enum_spec.values) <= _MAX_ENUM_VALUES
                and upper_camel_pattern.match(enum_spec.name)
            ):
                member_names = list({v.upper() for v in enum_spec.values})
                enum_type = Enum(enum_spec.name, member_names)
                output_fields[name] = (list[enum_type], Field(description=description))

            case FieldSpec(
                name=name, type="object", description=description, enum_spec=None, object_spec=object_spec
            ) if object_spec and upper_camel_pattern.match(object_spec.name):
                nested_model = _build_model(object_spec)
                output_fields[name] = (nested_model, Field(description=description))

            case FieldSpec(
                name=name, type="object_array", description=description, enum_spec=None, object_spec=object_spec
            ) if object_spec and upper_camel_pattern.match(object_spec.name):
                nested_model = _build_model(object_spec)
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
                object_spec=object_spec,
            ) if object_spec is not None:
                raise ValueError(
                    f"Field '{name}': enum type must not provide object_spec (got object_spec={object_spec!r})."
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
                object_spec=object_spec,
            ) if object_spec is not None:
                raise ValueError(
                    f"Field '{name}': enum_array type must not provide object_spec (got object_spec={object_spec!r})."
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
                object_spec=object_spec,
            ) if object_spec is not None and not upper_camel_pattern.match(object_spec.name):
                raise ValueError(
                    (
                        f"Field '{name}': object_spec.name '{object_spec.name}' must be UpperCamelCase "
                        "(regex ^[A-Z][A-Za-z0-9]*$) and contain only letters and digits."
                    )
                )
            # Object/object_array types must not provide enum_spec
            case FieldSpec(
                name=name,
                type="object" | "object_array",
                enum_spec=enum_spec,
                object_spec=object_spec,
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
                object_spec=object_spec,
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
                object_spec=object_spec,
            ) if object_spec is not None:
                raise ValueError(
                    (
                        f"Field '{name}': type '{field.type}' must not define object_spec "
                        f"(got object_spec={object_spec!r})."
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

    return create_model(object_spec.name, **output_fields)
