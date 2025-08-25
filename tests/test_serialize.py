from enum import Enum
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from openaivec._serialize import deserialize_base_model, serialize_base_model


class Gender(str, Enum):
    FEMALE = "FEMALE"
    MALE = "MALE"


class Person(BaseModel):
    name: str
    age: int
    gender: Gender


class Team(BaseModel):
    name: str
    members: list[Person]
    rules: list[str]


class Matrix(BaseModel):
    data: list[list[float]]


class ModelWithDescriptions(BaseModel):
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    confidence: float = Field(description="Confidence score for sentiment (0.0-1.0)")


class ModelWithoutDescriptions(BaseModel):
    name: str
    age: int


class MixedModel(BaseModel):
    name: str  # No description
    age: int = Field()  # Field() without description
    email: str = Field(description="User's email address")  # With description


class TaskStatus(BaseModel):
    status: Literal["pending", "in_progress", "completed"]
    priority: Literal["high", "medium", "low"]
    category: str = Field(description="Task category")


class ComplexLiteralModel(BaseModel):
    """Model with various Literal types including numbers and mixed types."""

    text_status: Literal["active", "inactive", "pending"]
    numeric_level: Literal[1, 2, 3, 4, 5]
    mixed_values: Literal["default", 42, True]
    optional_literal: Literal["yes", "no"] = "no"


class NestedLiteralModel(BaseModel):
    """Model with nested structures containing Literal types."""

    config: TaskStatus
    settings: list[Literal["debug", "info", "warning", "error"]]
    metadata: dict = Field(default_factory=dict)


class TestDeserialize:
    def test_deserialize(self):
        cls = deserialize_base_model(Team.model_json_schema())
        json_schema = cls.model_json_schema()
        assert json_schema["title"] == "Team"
        assert json_schema["type"] == "object"
        assert json_schema["properties"]["name"]["type"] == "string"
        assert json_schema["properties"]["members"]["type"] == "array"
        assert json_schema["properties"]["rules"]["type"] == "array"
        assert json_schema["properties"]["rules"]["items"]["type"] == "string"

    def test_deserialize_with_nested_list(self):
        cls = deserialize_base_model(Matrix.model_json_schema())
        json_schema = cls.model_json_schema()
        assert json_schema["title"] == "Matrix"
        assert json_schema["type"] == "object"
        assert json_schema["properties"]["data"]["type"] == "array"
        assert json_schema["properties"]["data"]["items"]["type"] == "array"
        assert json_schema["properties"]["data"]["items"]["items"]["type"] == "number"

    @pytest.mark.parametrize(
        "model_class,expected_descriptions,test_data,test_case",
        [
            (
                ModelWithDescriptions,
                {
                    "sentiment": "Overall sentiment: positive, negative, or neutral",
                    "confidence": "Confidence score for sentiment (0.0-1.0)",
                },
                {"sentiment": "positive", "confidence": 0.95},
                "with_descriptions",
            ),
            (
                ModelWithoutDescriptions,
                {"name": None, "age": None},
                {"name": "John", "age": 30},
                "without_descriptions",
            ),
            (
                MixedModel,
                {"name": None, "age": None, "email": "User's email address"},
                {"name": "Jane", "age": 25, "email": "jane@example.com"},
                "mixed_descriptions",
            ),
        ],
    )
    def test_field_descriptions_serialization(self, model_class, expected_descriptions, test_data, test_case):
        """Test that Field descriptions are preserved during serialization/deserialization."""
        # Serialize and deserialize
        serialized = serialize_base_model(model_class)
        deserialized = deserialize_base_model(serialized)
        deserialized_schema = deserialized.model_json_schema()

        # Check descriptions match expectations
        for field_name, expected_desc in expected_descriptions.items():
            actual_desc = deserialized_schema["properties"][field_name].get("description")
            assert actual_desc == expected_desc, f"Field '{field_name}' description mismatch in {test_case}"

        # Test that instances can be created with expected values
        instance = deserialized(**test_data)
        for field_name, expected_value in test_data.items():
            assert getattr(instance, field_name) == expected_value

    def test_literal_enum_serialization(self):
        """Test that Literal enum types are properly serialized to JSON schema."""
        schema = serialize_base_model(TaskStatus)

        # Check that Literal types are converted to enum in JSON schema
        assert schema["properties"]["status"]["type"] == "string"
        assert set(schema["properties"]["status"]["enum"]) == {"pending", "in_progress", "completed"}

        assert schema["properties"]["priority"]["type"] == "string"
        assert set(schema["properties"]["priority"]["enum"]) == {"high", "medium", "low"}

        # Check that description is preserved
        assert schema["properties"]["category"]["description"] == "Task category"

    def test_literal_enum_deserialization(self):
        """Test that Literal enum types are properly deserialized from JSON schema."""
        original_schema = serialize_base_model(TaskStatus)
        deserialized_class = deserialize_base_model(original_schema)

        # Test successful creation with valid values
        instance = deserialized_class(status="pending", priority="high", category="development")
        assert instance.status == "pending"
        assert instance.priority == "high"
        assert instance.category == "development"

        # Test validation with invalid values
        with pytest.raises(ValueError):
            deserialized_class(status="invalid_status", priority="high", category="development")

        with pytest.raises(ValueError):
            deserialized_class(status="pending", priority="invalid_priority", category="development")

    def test_complex_literal_types(self):
        """Test serialization/deserialization of complex Literal types with mixed values."""
        schema = serialize_base_model(ComplexLiteralModel)

        # Check text literals
        assert set(schema["properties"]["text_status"]["enum"]) == {"active", "inactive", "pending"}

        # Check numeric literals
        assert set(schema["properties"]["numeric_level"]["enum"]) == {1, 2, 3, 4, 5}

        # Check mixed type literals
        assert set(schema["properties"]["mixed_values"]["enum"]) == {"default", 42, True}

        # Check optional literal with default
        assert set(schema["properties"]["optional_literal"]["enum"]) == {"yes", "no"}

        # Test deserialization
        deserialized_class = deserialize_base_model(schema)

        # Test with valid values
        instance = deserialized_class(text_status="active", numeric_level=3, mixed_values="default")
        # String-only literals are stored as Literal values
        assert instance.text_status == "active"
        # Numeric and mixed types use Literal, so values are stored directly
        assert instance.numeric_level == 3
        assert instance.mixed_values == "default"
        assert instance.optional_literal == "no"  # default value

        # Test with mixed value types
        instance2 = deserialized_class(text_status="inactive", numeric_level=5, mixed_values=42, optional_literal="yes")
        assert instance2.text_status == "inactive"
        assert instance2.numeric_level == 5
        assert instance2.mixed_values == 42
        assert instance2.optional_literal == "yes"

    def test_nested_literal_models(self):
        """Test serialization/deserialization of nested models containing Literal types."""
        schema = serialize_base_model(NestedLiteralModel)

        # Check nested model structure
        assert "config" in schema["properties"]
        assert "settings" in schema["properties"]

        # Check array of literals
        assert schema["properties"]["settings"]["type"] == "array"
        assert set(schema["properties"]["settings"]["items"]["enum"]) == {"debug", "info", "warning", "error"}

        # Test deserialization
        deserialized_class = deserialize_base_model(schema)

        # Create nested instance
        instance = deserialized_class(
            config={"status": "completed", "priority": "medium", "category": "testing"},
            settings=["debug", "info"],
            metadata={"version": "1.0"},
        )

        assert instance.config.status == "completed"
        assert instance.config.priority == "medium"
        assert instance.config.category == "testing"
        # For list of literals, they are stored directly as values
        assert instance.settings == ["debug", "info"]
        assert instance.metadata == {"version": "1.0"}

    @pytest.mark.parametrize(
        "model_class,valid_data,invalid_field,invalid_value,test_case",
        [
            (
                TaskStatus,
                {"status": "pending", "priority": "high", "category": "test"},
                "status",
                "invalid",
                "TaskStatus",
            ),
            (
                ComplexLiteralModel,
                {"text_status": "active", "numeric_level": 3, "mixed_values": "default"},
                "text_status",
                "invalid",
                "ComplexLiteralModel",
            ),
        ],
    )
    def test_literal_roundtrip_consistency(self, model_class, valid_data, invalid_field, invalid_value, test_case):
        """Test that Literal types maintain consistency through serialize/deserialize cycles."""
        # Serialize original model
        original_schema = serialize_base_model(model_class)

        # Deserialize to get new class
        deserialized_class = deserialize_base_model(original_schema)

        # Test that the deserialized class can create valid instances
        instance = deserialized_class(**valid_data)
        for field_name, expected_value in valid_data.items():
            assert getattr(instance, field_name) == expected_value

        # Test validation with invalid data
        invalid_data = valid_data.copy()
        invalid_data[invalid_field] = invalid_value
        with pytest.raises(ValueError):
            deserialized_class(**invalid_data)
