"""Refactored serialization utilities for Pydantic BaseModel classes.

This module provides utilities for converting Pydantic BaseModel classes
to and from JSON schema representations with simplified, maintainable code.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

__all__ = []


def serialize_base_model(obj: type[BaseModel]) -> dict[str, Any]:
    """Serialize a Pydantic BaseModel to JSON schema."""
    return obj.model_json_schema()


def dereference_json_schema(json_schema: dict[str, Any]) -> dict[str, Any]:
    """Dereference JSON schema by resolving $ref pointers with circular reference protection."""
    model_map = json_schema.get("$defs", {})

    def dereference(obj, current_path=None):
        if current_path is None:
            current_path = []

        if isinstance(obj, dict):
            if "$ref" in obj:
                ref = obj["$ref"].split("/")[-1]

                # Check for circular reference
                if ref in current_path:
                    # Return a placeholder to break the cycle
                    return {"type": "object", "description": f"Circular reference to {ref}"}

                if ref in model_map:
                    # Add to path and recurse
                    new_path = current_path + [ref]
                    return dereference(model_map[ref], new_path)
                else:
                    # Invalid reference, return placeholder
                    return {"type": "object", "description": f"Invalid reference to {ref}"}
            else:
                return {k: dereference(v, current_path) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [dereference(x, current_path) for x in obj]
        else:
            return obj

    result = {}
    for k, v in json_schema.items():
        if k == "$defs":
            continue
        result[k] = dereference(v)

    return result


# ============================================================================
# Type Resolution - Separated into focused functions
# ============================================================================


def _resolve_union_type(union_options: list[dict[str, Any]]) -> type:
    """Resolve anyOf/oneOf to Union type."""
    union_types = []
    for option in union_options:
        if option.get("type") == "null":
            union_types.append(type(None))
        else:
            union_types.append(parse_field(option))

    if len(union_types) == 1:
        return union_types[0]
    elif len(union_types) == 2 and type(None) in union_types:
        # Optional type: T | None
        non_none_type = next(t for t in union_types if t is not type(None))
        return non_none_type | None  # type: ignore[return-value]
    else:
        from typing import Union

        return Union[tuple(union_types)]  # type: ignore[return-value]


def _resolve_basic_type(type_name: str, field_def: dict[str, Any]) -> type:
    """Resolve basic JSON schema types to Python types."""
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "null": type(None),
    }

    if type_name in type_mapping:
        return type_mapping[type_name]  # type: ignore[return-value]
    elif type_name == "object":
        # Check if it's a nested model or generic dict
        if "properties" in field_def:
            return deserialize_base_model(field_def)
        else:
            return dict
    elif type_name == "array":
        if "items" in field_def:
            inner_type = parse_field(field_def["items"])
            return list[inner_type]
        else:
            return list[Any]
    else:
        raise ValueError(f"Unsupported type: {type_name}")


def parse_field(field_def: dict[str, Any]) -> type:
    """Parse a JSON schema field definition to a Python type.

    Simplified version with clear separation of concerns.
    """
    # Handle union types
    if "anyOf" in field_def:
        return _resolve_union_type(field_def["anyOf"])
    if "oneOf" in field_def:
        return _resolve_union_type(field_def["oneOf"])

    # Handle basic types
    if "type" not in field_def:
        return Any  # type: ignore[return-value]

    return _resolve_basic_type(field_def["type"], field_def)


# ============================================================================
# Field Information Creation - Centralized logic
# ============================================================================


def _create_field_info(description: str | None, default_value: Any, is_required: bool) -> Field:  # type: ignore[type-arg]
    """Create Field info with consistent logic."""
    if is_required and default_value is None:
        # Required field without default
        return Field(description=description) if description else Field()
    else:
        # Optional field or field with default
        return Field(default=default_value, description=description) if description else Field(default=default_value)


def _make_optional_if_needed(field_type: type, is_required: bool, has_default: bool) -> type:
    """Make field type optional if needed."""
    if is_required or has_default:
        return field_type

    # Check if already nullable
    from typing import Union

    if hasattr(field_type, "__origin__") and field_type.__origin__ is Union and type(None) in field_type.__args__:
        return field_type

    # Make optional
    return field_type | None  # type: ignore[return-value]


# ============================================================================
# Field Processing - Separated enum and regular field logic
# ============================================================================


def _process_enum_field(field_name: str, field_def: dict[str, Any], is_required: bool) -> tuple[type, Field]:  # type: ignore[type-arg]
    """Process enum field with Literal type."""
    enum_values = field_def["enum"]

    # Create Literal type
    if len(enum_values) == 1:
        literal_type = Literal[enum_values[0]]
    else:
        literal_type = Literal[tuple(enum_values)]

    # Handle optionality
    description = field_def.get("description")
    default_value = field_def.get("default")
    has_default = default_value is not None

    if not is_required and not has_default:
        literal_type = literal_type | None  # type: ignore[assignment]
        default_value = None

    field_info = _create_field_info(description, default_value, is_required)
    return literal_type, field_info  # type: ignore[return-value]


def _process_regular_field(field_name: str, field_def: dict[str, Any], is_required: bool) -> tuple[type, Field]:  # type: ignore[type-arg]
    """Process regular (non-enum) field."""
    field_type = parse_field(field_def)
    description = field_def.get("description")
    default_value = field_def.get("default")
    has_default = default_value is not None

    # Handle optionality
    field_type = _make_optional_if_needed(field_type, is_required, has_default)

    if not is_required and not has_default:
        default_value = None

    field_info = _create_field_info(description, default_value, is_required)
    return field_type, field_info


# ============================================================================
# Main Schema Processing - Clean and focused
# ============================================================================


def deserialize_base_model(json_schema: dict[str, Any]) -> type[BaseModel]:
    """Deserialize a JSON schema to a Pydantic BaseModel class.

    Refactored version with clear separation of concerns and simplified logic.
    """
    # Basic setup
    title = json_schema.get("title", "DynamicModel")
    dereferenced_schema = dereference_json_schema(json_schema)
    properties = dereferenced_schema.get("properties", {})
    required_fields = set(dereferenced_schema.get("required", []))

    # Process each field
    fields = {}
    for field_name, field_def in properties.items():
        is_required = field_name in required_fields

        if "enum" in field_def:
            field_type, field_info = _process_enum_field(field_name, field_def, is_required)
        else:
            field_type, field_info = _process_regular_field(field_name, field_def, is_required)

        fields[field_name] = (field_type, field_info)

    return create_model(title, **fields)
