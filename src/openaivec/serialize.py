"""Serialization utilities for Pydantic BaseModel classes.

This module provides utilities for converting Pydantic BaseModel classes
to and from JSON schema representations. It supports dynamic model creation
from JSON schemas with special handling for enum fields, which are converted
to Literal types for better type safety and compatibility.

Example:
    Basic serialization and deserialization:
    
    ```python
    from pydantic import BaseModel
    from typing import Literal
    
    class Status(BaseModel):
        value: Literal["active", "inactive"]
        description: str
    
    # Serialize to JSON schema
    schema = serialize_base_model(Status)
    
    # Deserialize back to BaseModel class
    DynamicStatus = deserialize_base_model(schema)
    instance = DynamicStatus(value="active", description="User is active")
    ```
"""

from typing import Any, Dict, List, Type, Literal

from pydantic import BaseModel, Field, create_model

__all__ = ["deserialize_base_model", "serialize_base_model"]


def serialize_base_model(obj: Type[BaseModel]) -> Dict[str, Any]:
    """Serialize a Pydantic BaseModel to JSON schema.
    
    Args:
        obj: The Pydantic BaseModel class to serialize.
        
    Returns:
        A dictionary containing the JSON schema representation of the model.
        
    Example:
        ```python
        from pydantic import BaseModel
        
        class Person(BaseModel):
            name: str
            age: int
            
        schema = serialize_base_model(Person)
        ```
    """
    return obj.model_json_schema()


def dereference_json_schema(json_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Dereference JSON schema by resolving $ref pointers.
    
    This function resolves all $ref references in a JSON schema by replacing
    them with the actual referenced definitions from the $defs section.
    
    Args:
        json_schema: The JSON schema containing potential $ref references.
        
    Returns:
        A dereferenced JSON schema with all $ref pointers resolved.
        
    Example:
        ```python
        schema = {
            "properties": {
                "user": {"$ref": "#/$defs/User"}
            },
            "$defs": {
                "User": {"type": "object", "properties": {"name": {"type": "string"}}}
            }
        }
        dereferenced = dereference_json_schema(schema)
        # user property will contain the actual User definition
        ```
    """
    model_map = json_schema.get("$defs", {})

    def dereference(obj):
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref = obj["$ref"].split("/")[-1]
                return dereference(model_map[ref])
            else:
                return {k: dereference(v) for k, v in obj.items()}

        elif isinstance(obj, list):
            return [dereference(x) for x in obj]
        else:
            return obj

    result = {}
    for k, v in json_schema.items():
        if k == "$defs":
            continue

        result[k] = dereference(v)

    return result


def parse_field(v: Dict[str, Any]) -> Any:
    """Parse a JSON schema field definition to a Python type.
    
    Converts JSON schema field definitions to corresponding Python types
    for use in Pydantic model creation.
    
    Args:
        v: A dictionary containing the JSON schema field definition.
        
    Returns:
        The corresponding Python type (str, int, float, bool, dict, List, or BaseModel).
        
    Raises:
        ValueError: If the field type is not supported.
        
    Example:
        ```python
        field_def = {"type": "string"}
        python_type = parse_field(field_def)  # Returns str
        
        array_def = {"type": "array", "items": {"type": "integer"}}
        python_type = parse_field(array_def)  # Returns List[int]
        ```
    """
    t = v["type"]
    if t == "string":
        return str
    elif t == "integer":
        return int
    elif t == "number":
        return float
    elif t == "boolean":
        return bool
    elif t == "object":
        # Check if it's a generic object (dict) or a nested model
        if "properties" in v:
            return deserialize_base_model(v)
        else:
            return dict
    elif t == "array":
        inner_type = parse_field(v["items"])
        return List[inner_type]
    else:
        raise ValueError(f"Unsupported type: {t}")


def deserialize_base_model(json_schema: Dict[str, Any]) -> Type[BaseModel]:
    """Deserialize a JSON schema to a Pydantic BaseModel class.
    
    Creates a dynamic Pydantic BaseModel class from a JSON schema definition.
    For enum fields, this function uses Literal types instead of Enum classes
    for better type safety and compatibility with systems like Apache Spark.
    
    Args:
        json_schema: A dictionary containing the JSON schema definition.
        
    Returns:
        A dynamically created Pydantic BaseModel class.
        
    Example:
        ```python
        schema = {
            "title": "Person",
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive"],
                    "description": "Person's status"
                }
            }
        }
        
        PersonModel = deserialize_base_model(schema)
        person = PersonModel(name="John", status="active")
        ```
        
    Note:
        Enum fields are converted to Literal types for improved compatibility
        and type safety. This ensures better integration with data processing
        frameworks like Apache Spark.
    """
    fields = {}
    properties = dereference_json_schema(json_schema).get("properties", {})

    for k, v in properties.items():
        if "enum" in v:
            enum_values = v["enum"]
            
            # Always use Literal instead of Enum for better type safety and Spark compatibility
            if len(enum_values) == 1:
                literal_type = Literal[enum_values[0]]
            else:
                # Create Literal with multiple values
                literal_type = Literal[tuple(enum_values)]
            
            description = v.get("description")
            default_value = v.get("default")
            
            if default_value is not None:
                field_info = Field(default=default_value, description=description) if description is not None else Field(default=default_value)
            else:
                field_info = Field(description=description) if description is not None else Field()
            
            fields[k] = (literal_type, field_info)
        else:
            description = v.get("description")
            default_value = v.get("default")
            
            if default_value is not None:
                field_info = Field(default=default_value, description=description) if description is not None else Field(default=default_value)
            else:
                field_info = Field(description=description) if description is not None else Field()
            
            fields[k] = (parse_field(v), field_info)
    return create_model(json_schema["title"], **fields)
