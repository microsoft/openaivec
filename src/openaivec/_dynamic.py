from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, create_model

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
            "homogeneous list of the base type. 'enum' / 'enum_array' require 'enum_values'. 'object' / 'object_array' "
            "require 'object_spec'. Primitives must not define 'enum_values' or 'object_spec'."
        )
    )
    description: str = Field(
        description=(
            "Human‑readable, concise explanation of the field's meaning and business intent. Should clarify units, "
            "value semantics, and any domain constraints not captured by type. 1–2 sentences; no implementation notes."
        )
    )
    enum_values: list[str] | None = Field(
        default=None,
        description=(
            "List of allowed case-insensitive symbols for 'enum' / 'enum_array'. Required and non-empty for those "
            f"types; must be omitted for all others. Maximum {_MAX_ENUM_VALUES} distinct values (after dedup). "
            "Duplicates are "
            "ignored (deduplicated before Enum creation)."
        ),
    )
    object_spec: ObjectSpec | None = Field(
        default=None,
        description=(
            "Nested object schema. Required for 'object' / 'object_array'; must be omitted for every other type. The "
            "contained 'name' is used to derive the generated nested Pydantic model class name."
        ),
    )


class ObjectSpec(BaseModel):
    name: str = Field(
        description=(
            "Object name in lower_snake_case (singular noun). Used to derive the generated Pydantic model class name "
            "(UpperCamelCase). Must start with a lowercase letter; only lowercase letters, numbers, underscores "
            "allowed."
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

        # Convert field name to UpperCamelCase for class attribute names
        upper_camel_field_name: str = "".join([w.capitalize() for w in field.name.split("_")])
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
                enum_values=None,
                object_spec=None,
            ):
                field_type = type_map[field.type]
                output_fields[name] = (field_type, Field(description=description))

            case FieldSpec(
                name=name, type="enum", description=description, enum_values=enum_values, object_spec=None
            ) if enum_values and len(enum_values) <= _MAX_ENUM_VALUES:
                unique_members: list[str] = [v.upper() for v in set(enum_values)]

                enum_type = Enum(upper_camel_field_name, unique_members)
                output_fields[name] = (enum_type, Field(description=description))

            case FieldSpec(
                name=name, type="enum_array", description=description, enum_values=enum_values, object_spec=None
            ) if enum_values and len(enum_values) <= _MAX_ENUM_VALUES:
                unique_members: list[str] = [v.upper() for v in set(enum_values)]

                enum_type = Enum(upper_camel_field_name, unique_members)
                output_fields[name] = (list[enum_type], Field(description=description))

            case FieldSpec(
                name=name, type="object", description=description, enum_values=None, object_spec=object_spec
            ) if object_spec:
                nested_model = _build_model(object_spec)
                output_fields[name] = (nested_model, Field(description=description))

            case FieldSpec(
                name=name, type="object_array", description=description, enum_values=None, object_spec=object_spec
            ) if object_spec:
                nested_model = _build_model(object_spec)
                output_fields[name] = (list[nested_model], Field(description=description))

            # ---- Error cases (explicit reasons) ----
            # Enum type without enum_values (None or empty)
            case FieldSpec(
                name=name,
                type="enum",
                enum_values=enum_values,
                object_spec=None,
            ) if not enum_values:
                raise ValueError(
                    f"Field '{name}': enum type requires non-empty enum_values list (got {enum_values!r})."
                )
            # Enum type exceeding max length
            case FieldSpec(
                name=name,
                type="enum",
                enum_values=enum_values,
                object_spec=None,
            ) if enum_values and len(enum_values) > _MAX_ENUM_VALUES:
                raise ValueError(
                    (
                        f"Field '{name}': enum type supports at most {_MAX_ENUM_VALUES} enum_values "
                        f"(got {len(enum_values)})."
                    )
                )
            # Enum type incorrectly provides an object_spec
            case FieldSpec(
                name=name,
                type="enum",
                enum_values=enum_values,
                object_spec=object_spec,
            ) if object_spec is not None:
                raise ValueError(
                    f"Field '{name}': enum type must not provide object_spec (got object_spec={object_spec!r})."
                )
            # Enum array type without enum_values
            case FieldSpec(
                name=name,
                type="enum_array",
                enum_values=enum_values,
                object_spec=None,
            ) if not enum_values:
                raise ValueError(
                    f"Field '{name}': enum_array type requires non-empty enum_values list (got {enum_values!r})."
                )
            # Enum array type exceeding max length
            case FieldSpec(
                name=name,
                type="enum_array",
                enum_values=enum_values,
                object_spec=None,
            ) if enum_values and len(enum_values) > _MAX_ENUM_VALUES:
                raise ValueError(
                    (
                        f"Field '{name}': enum_array type supports at most {_MAX_ENUM_VALUES} enum_values "
                        f"(got {len(enum_values)})."
                    )
                )
            # Enum array type incorrectly provides an object_spec
            case FieldSpec(
                name=name,
                type="enum_array",
                enum_values=enum_values,
                object_spec=object_spec,
            ) if object_spec is not None:
                raise ValueError(
                    f"Field '{name}': enum_array type must not provide object_spec (got object_spec={object_spec!r})."
                )
            # Object type missing object_spec
            case FieldSpec(
                name=name,
                type="object",
                enum_values=enum_values,
                object_spec=None,
            ):
                raise ValueError(f"Field '{name}': object type requires object_spec (got object_spec=None).")
            # Object array type missing object_spec
            case FieldSpec(
                name=name,
                type="object_array",
                enum_values=enum_values,
                object_spec=None,
            ):
                raise ValueError(f"Field '{name}': object_array type requires object_spec (got object_spec=None).")
            # Object/object_array types must not provide enum_values
            case FieldSpec(
                name=name,
                type="object" | "object_array",
                enum_values=enum_values,
                object_spec=object_spec,
            ) if enum_values is not None:
                raise ValueError(
                    f"Field '{name}': {field.type} must not define enum_values (got enum_values={enum_values!r})."
                )
            # Primitive / simple array types must not have enum_values
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
                enum_values=enum_values,
                object_spec=object_spec,
            ) if enum_values is not None:
                raise ValueError(
                    (
                        f"Field '{name}': type '{field.type}' must not define enum_values "
                        f"(got enum_values={enum_values!r})."
                    )
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
                enum_values=None,
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
                        f"name={f.name!r}, type={f.type!r}, enum_values={f.enum_values!r}, "
                        f"object_spec={'set' if f.object_spec else None}."
                    )
                )

    upper_camel_object_name: str = "".join([w.capitalize() for w in object_spec.name.split("_")])
    return create_model(upper_camel_object_name, **output_fields)
