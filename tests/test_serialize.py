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

    def test_field_descriptions_preserved(self):
        """Test that Field descriptions are preserved during serialization/deserialization."""
        original = ModelWithDescriptions
        original_schema = original.model_json_schema()

        # Serialize and deserialize
        serialized = serialize_base_model(original)
        deserialized = deserialize_base_model(serialized)
        deserialized_schema = deserialized.model_json_schema()

        # Check that descriptions are preserved
        original_sentiment_desc = original_schema["properties"]["sentiment"].get("description")
        deserialized_sentiment_desc = deserialized_schema["properties"]["sentiment"].get("description")
        assert original_sentiment_desc == deserialized_sentiment_desc

        original_confidence_desc = original_schema["properties"]["confidence"].get("description")
        deserialized_confidence_desc = deserialized_schema["properties"]["confidence"].get("description")
        assert original_confidence_desc == deserialized_confidence_desc

        # Test that instances can be created
        instance = deserialized(sentiment="positive", confidence=0.95)
        assert instance.sentiment == "positive"
        assert instance.confidence == 0.95

    def test_model_without_descriptions(self):
        """Test that models without Field descriptions work correctly."""
        original = ModelWithoutDescriptions

        # Serialize and deserialize
        serialized = serialize_base_model(original)
        deserialized = deserialize_base_model(serialized)
        deserialized_schema = deserialized.model_json_schema()

        # Check that no descriptions are present (should be None or absent)
        assert deserialized_schema["properties"]["name"].get("description") is None
        assert deserialized_schema["properties"]["age"].get("description") is None

        # Test that instances can be created
        instance = deserialized(name="John", age=30)
        assert instance.name == "John"
        assert instance.age == 30

    def test_mixed_descriptions(self):
        """Test models with mixed field descriptions (some with, some without)."""
        original = MixedModel

        # Serialize and deserialize
        serialized = serialize_base_model(original)
        deserialized = deserialize_base_model(serialized)
        deserialized_schema = deserialized.model_json_schema()

        # Check mixed descriptions
        assert deserialized_schema["properties"]["name"].get("description") is None
        assert deserialized_schema["properties"]["age"].get("description") is None
        assert deserialized_schema["properties"]["email"].get("description") == "User's email address"

        # Test that instances can be created
        instance = deserialized(name="Jane", age=25, email="jane@example.com")
        assert instance.name == "Jane"
        assert instance.age == 25
        assert instance.email == "jane@example.com"

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

    def test_literal_roundtrip_consistency(self):
        """Test that Literal types maintain consistency through serialize/deserialize cycles."""
        models_to_test = [TaskStatus, ComplexLiteralModel]

        for model_class in models_to_test:
            # Serialize original model
            original_schema = serialize_base_model(model_class)

            # Deserialize to get new class
            deserialized_class = deserialize_base_model(original_schema)

            # Test that the deserialized class can create valid instances
            # and that enum validation still works
            if model_class == TaskStatus:
                # Test TaskStatus functionality
                instance = deserialized_class(status="pending", priority="high", category="test")
                assert instance.status == "pending"
                assert instance.priority == "high"
                assert instance.category == "test"

                # Test validation
                with pytest.raises(ValueError):
                    deserialized_class(status="invalid", priority="high", category="test")

            elif model_class == ComplexLiteralModel:
                # Test ComplexLiteralModel functionality
                instance = deserialized_class(text_status="active", numeric_level=3, mixed_values="default")
                assert instance.text_status == "active"
                assert instance.numeric_level == 3
                assert instance.mixed_values == "default"

                # Test validation
                with pytest.raises(ValueError):
                    deserialized_class(text_status="invalid", numeric_level=3, mixed_values="default")
