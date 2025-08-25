"""Test serialization with Pydantic v2 compliance."""

from pydantic import BaseModel, Field

from openaivec._serialize import deserialize_base_model, serialize_base_model


class TestSerializationPydanticV2:
    def test_pydantic_v2_api(self):
        """Test that the serialization/deserialization works with Pydantic v2 API."""

        class TestModel(BaseModel):
            name: str = Field(description="The name of the item")
            value: float = Field(description="The value")

        # Test serialization
        schema = serialize_base_model(TestModel)
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "value" in schema["properties"]

        # Test deserialization
        reconstructed = deserialize_base_model(schema)

        # Test that the reconstructed class works
        instance = reconstructed(name="test", value=3.14)
        assert instance.name == "test"
        assert instance.value == 3.14

        # Test that field descriptions are preserved
        reconstructed_schema = reconstructed.model_json_schema()
        assert reconstructed_schema["properties"]["name"]["description"] == "The name of the item"
        assert reconstructed_schema["properties"]["value"]["description"] == "The value"

    def test_model_fields_api(self):
        """Test using Pydantic v2's model_fields API."""

        class ModelWithDefaults(BaseModel):
            required_field: str
            optional_field: str = "default_value"
            field_with_description: int = Field(default=42, description="An integer field")

        # Serialize and deserialize
        schema = serialize_base_model(ModelWithDefaults)
        reconstructed = deserialize_base_model(schema)

        # Test required fields work
        instance = reconstructed(required_field="test")
        assert instance.required_field == "test"
        assert instance.optional_field == "default_value"
        assert instance.field_with_description == 42

        # Test the field with description
        reconstructed_schema = reconstructed.model_json_schema()
        assert reconstructed_schema["properties"]["field_with_description"]["description"] == "An integer field"

    def test_model_json_schema_api(self):
        """Test using Pydantic v2's model_json_schema API."""

        class AdvancedModel(BaseModel):
            """A model with various field types."""

            text: str = Field(title="Text Field", description="A text field")
            number: int = Field(ge=0, le=100, description="A number between 0 and 100")
            optional_bool: bool = True

        # Get the JSON schema using v2 API
        original_schema = AdvancedModel.model_json_schema()

        # Serialize using our function
        our_schema = serialize_base_model(AdvancedModel)

        # They should be the same
        assert our_schema == original_schema

        # Test round-trip
        reconstructed = deserialize_base_model(our_schema)
        reconstructed_schema = reconstructed.model_json_schema()

        # Key properties should be preserved
        assert reconstructed_schema["properties"]["text"]["description"] == "A text field"
        assert reconstructed_schema["properties"]["number"]["description"] == "A number between 0 and 100"

    def test_nested_models_v2(self):
        """Test nested models with Pydantic v2."""

        class Address(BaseModel):
            street: str = Field(description="Street address")
            city: str = Field(description="City name")
            zipcode: str = Field(description="ZIP code")

        class Person(BaseModel):
            name: str = Field(description="Person's full name")
            age: int = Field(description="Person's age")
            address: Address = Field(description="Person's address")

        # Serialize and deserialize
        schema = serialize_base_model(Person)
        reconstructed = deserialize_base_model(schema)

        # Test creating instances
        address_data = {"street": "123 Main St", "city": "Anytown", "zipcode": "12345"}
        person_data = {"name": "John Doe", "age": 30, "address": address_data}

        instance = reconstructed(**person_data)
        assert instance.name == "John Doe"
        assert instance.age == 30
        assert instance.address.street == "123 Main St"
        assert instance.address.city == "Anytown"
        assert instance.address.zipcode == "12345"

        # Test that descriptions are preserved
        reconstructed_schema = reconstructed.model_json_schema()
        assert reconstructed_schema["properties"]["name"]["description"] == "Person's full name"
        # Note: Nested model field descriptions are not preserved in the current implementation
        # This is a known limitation when using $ref with Pydantic schema serialization
        assert "$ref" in reconstructed_schema["properties"]["address"]
