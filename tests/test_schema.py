from enum import Enum
from typing import get_args, get_origin

import pytest
from pydantic import BaseModel

from openaivec._dynamic import EnumSpec, FieldSpec, ObjectSpec  # internal types for constructing test schemas
from openaivec._schema import InferredSchema, SchemaInferenceInput, SchemaInferer  # type: ignore


@pytest.fixture(scope="session")
def cached_inferred_schemas(openai_client, responses_model_name):
    """Cache expensive schema inference operations for the entire test session."""
    datasets = {
        "basic_support": SchemaInferenceInput(
            examples=[
                "Order #1234: customer requested refund due to damaged packaging.",
                "Order #1235: customer happy, praised fast shipping.",
                "Order #1236: delayed delivery complaint, wants status update.",
            ],
            instructions="Extract useful flat analytic signals from short support notes.",
        ),
        "retry_case": SchemaInferenceInput(
            examples=[
                "User reported login failure after password reset.",
                "User confirmed issue was resolved after cache clear.",
            ],
            instructions="Infer minimal status/phase signals from event style notes.",
        ),
    }

    inferer = SchemaInferer(client=openai_client, model_name=responses_model_name)
    inferred = {}
    for name, ds in datasets.items():
        inferred[name] = inferer.infer_schema(ds, max_retries=2)

    return inferred, datasets


@pytest.mark.requires_api
@pytest.mark.slow
class TestSchemaInferer:
    @pytest.fixture(autouse=True)
    def setup_cached_data(self, cached_inferred_schemas):
        """Setup cached inference data for all tests."""
        self.inferred_schemas, self.datasets = cached_inferred_schemas

    def test_inference_basic(self):
        allowed_types = {
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
        }
        for inferred in self.inferred_schemas.values():
            assert isinstance(inferred.object_spec, ObjectSpec)
            assert isinstance(inferred.object_spec.fields, list)
            assert len(inferred.object_spec.fields) >= 0
            for f in inferred.object_spec.fields:
                assert f.type in allowed_types
                if f.type in {"enum", "enum_array"}:
                    assert f.enum_spec is not None
                    assert len(f.enum_spec.values) > 0
                    assert len(f.enum_spec.values) <= 24
                else:
                    assert f.enum_spec is None

    def test_build_model(self):
        inferred = self.inferred_schemas["basic_support"]
        model_cls = inferred.build_model()
        assert issubclass(model_cls, BaseModel)
        props = model_cls.model_json_schema().get("properties", {})
        assert props

    @pytest.mark.slow
    def test_retry(self, openai_client, responses_model_name):
        # Simple retry test without complex mocking
        # Just verify the retry functionality works with actual API calls
        ds = self.datasets["retry_case"]
        inferer = SchemaInferer(client=openai_client, model_name=responses_model_name)
        suggestion = inferer.infer_schema(ds, max_retries=3)

        # Verify the suggestion is valid
        assert isinstance(suggestion.object_spec, ObjectSpec)
        assert len(suggestion.object_spec.fields) >= 0
        for f in suggestion.object_spec.fields:
            if f.type in {"enum", "enum_array"}:
                assert f.enum_spec is not None
                assert len(f.enum_spec.values) > 0
                assert len(f.enum_spec.values) <= 24

    @pytest.mark.slow
    def test_structuring_basic(self, openai_client, responses_model_name):
        inferred = self.inferred_schemas["basic_support"]
        raw = self.datasets["basic_support"].examples[0]
        model_cls = inferred.build_model()
        parsed = openai_client.responses.parse(
            model=responses_model_name,
            instructions=inferred.inference_prompt,
            input=raw,
            text_format=model_cls,
        )
        structured = parsed.output_parsed
        assert isinstance(structured, BaseModel)

    def test_field_descriptions_in_model(self):
        """Test that field descriptions from FieldSpec are reflected in generated Pydantic model."""
        inferred = self.inferred_schemas["basic_support"]
        model_cls = inferred.build_model()
        # Get the model schema which includes field descriptions
        schema_json = model_cls.model_json_schema()
        properties = schema_json.get("properties", {})

        # Verify that all fields from the inferred schema have descriptions in the model
        for field_spec in inferred.object_spec.fields:
            field_name = field_spec.name
            assert field_name in properties, f"Field '{field_name}' should be in model properties"

            field_schema = properties[field_name]
            assert "description" in field_schema, f"Field '{field_name}' should have a description"
            assert field_schema["description"] == field_spec.description, (
                f"Field '{field_name}' description should match FieldSpec description"
            )


class TestInferredSchemaBuildModel:
    """Comprehensive MECE test cases for InferredSchema.build_model method."""

    def test_build_model_primitive_types(self):
        """Test that all primitive types are correctly mapped to Python types."""
        schema = InferredSchema(
            instructions="Test primitive types",
            examples_summary="Various primitive type examples",
            examples_instructions_alignment="Primitive examples justify coverage of all base types",
            object_spec=ObjectSpec(
                name="PrimitiveRoot",
                fields=[
                    FieldSpec(name="text_field", type="string", description="A string field"),
                    FieldSpec(name="number_field", type="integer", description="An integer field"),
                    FieldSpec(name="decimal_field", type="float", description="A float field"),
                    FieldSpec(name="flag_field", type="boolean", description="A boolean field"),
                ],
            ),
            inference_prompt="Test prompt",
        )

        model_cls = schema.build_model()
        schema_dict = model_cls.model_json_schema()
        properties = schema_dict["properties"]

        # Verify correct type mapping
        assert properties["text_field"]["type"] == "string"
        assert properties["number_field"]["type"] == "integer"
        assert properties["decimal_field"]["type"] == "number"
        assert properties["flag_field"]["type"] == "boolean"

        # Verify all fields are required
        assert set(schema_dict["required"]) == {"text_field", "number_field", "decimal_field", "flag_field"}

    def test_build_model_enum_field(self):
        """Test that enum fields generate proper Enum classes."""
        schema = InferredSchema(
            instructions="Test enum types",
            examples_summary="Enum examples",
            examples_instructions_alignment="Stable status labels appear repeatedly, supporting enum creation",
            object_spec=ObjectSpec(
                name="EnumRoot",
                fields=[
                    FieldSpec(
                        name="status_field",
                        type="enum",
                        description="Status enum field",
                        enum_spec=EnumSpec(name="Status", values=["active", "inactive", "pending"]),
                    ),
                    FieldSpec(name="regular_field", type="string", description="Regular string field"),
                ],
            ),
            inference_prompt="Test prompt",
        )

        model_cls = schema.build_model()

        # Verify enum field type
        status_annotation = model_cls.model_fields["status_field"].annotation
        assert issubclass(status_annotation, Enum)
        # Verify enum member names (uppercased unique set)
        member_names = {member.name for member in status_annotation}
        assert member_names == {"ACTIVE", "INACTIVE", "PENDING"}

        # Verify non-enum field is still string
        regular_annotation = model_cls.model_fields["regular_field"].annotation
        assert regular_annotation is str

    def test_build_model_field_ordering(self):
        """Test that field ordering is preserved in the generated model."""
        fields = [
            FieldSpec(name="third_field", type="string", description="Third field"),
            FieldSpec(name="first_field", type="integer", description="First field"),
            FieldSpec(name="second_field", type="boolean", description="Second field"),
        ]

        schema = InferredSchema(
            instructions="Test field ordering",
            examples_summary="Field ordering examples",
            examples_instructions_alignment="Ordering matters for deterministic downstream column alignment",
            object_spec=ObjectSpec(name="OrderingRoot", fields=fields),
            inference_prompt="Test prompt",
        )

        model_cls = schema.build_model()
        model_field_names = list(model_cls.model_fields.keys())
        expected_order = ["third_field", "first_field", "second_field"]

        assert model_field_names == expected_order

    def test_build_model_field_descriptions(self):
        """Test that field descriptions are correctly included in the model."""
        schema = InferredSchema(
            instructions="Test field descriptions",
            examples_summary="Description examples",
            examples_instructions_alignment="Descriptions guide extraction disambiguation",
            object_spec=ObjectSpec(
                name="DescRoot",
                fields=[
                    FieldSpec(name="described_field", type="string", description="This is a detailed description"),
                    FieldSpec(name="another_field", type="integer", description="Another detailed description"),
                ],
            ),
            inference_prompt="Test prompt",
        )

        model_cls = schema.build_model()
        schema_dict = model_cls.model_json_schema()
        properties = schema_dict["properties"]

        assert properties["described_field"]["description"] == "This is a detailed description"
        assert properties["another_field"]["description"] == "Another detailed description"

    def test_build_model_empty_fields(self):
        """Test behavior with empty fields list."""
        schema = InferredSchema(
            instructions="Test empty fields",
            examples_summary="Empty examples",
            examples_instructions_alignment="Edge case of no extractable signals",
            object_spec=ObjectSpec(name="EmptyRoot", fields=[]),
            inference_prompt="Test prompt",
        )

        model_cls = schema.build_model()
        assert len(model_cls.model_fields) == 0

        # Should still be a valid BaseModel
        assert issubclass(model_cls, BaseModel)

        # Should be able to instantiate with no arguments
        instance = model_cls()
        assert isinstance(instance, BaseModel)

    def test_build_model_mixed_enum_and_regular_fields(self):
        """Test a complex scenario with both enum and regular fields of all types."""
        schema = InferredSchema(
            instructions="Test mixed field types",
            examples_summary="Mixed type examples",
            examples_instructions_alignment="Examples demonstrate diverse field types including enums",
            object_spec=ObjectSpec(
                name="MixedRoot",
                fields=[
                    FieldSpec(
                        name="priority",
                        type="enum",
                        description="Priority level",
                        enum_spec=EnumSpec(name="Priority", values=["high", "medium", "low"]),
                    ),
                    FieldSpec(name="count", type="integer", description="Item count"),
                    FieldSpec(name="score", type="float", description="Quality score"),
                    FieldSpec(name="is_active", type="boolean", description="Active status"),
                    FieldSpec(
                        name="category",
                        type="enum",
                        description="Category name",
                        enum_spec=EnumSpec(name="Category", values=["A", "B", "C"]),
                    ),
                    FieldSpec(name="description", type="string", description="Free text description"),
                ],
            ),
            inference_prompt="Test prompt",
        )

        model_cls = schema.build_model()

        # Verify enum fields
        priority_type = model_cls.model_fields["priority"].annotation
        category_type = model_cls.model_fields["category"].annotation
        assert issubclass(priority_type, Enum)
        assert issubclass(category_type, Enum)

        # Verify regular fields
        assert model_cls.model_fields["count"].annotation is int
        assert model_cls.model_fields["score"].annotation is float
        assert model_cls.model_fields["is_active"].annotation is bool
        assert model_cls.model_fields["description"].annotation is str

        # Verify all fields are present
        assert len(model_cls.model_fields) == 6

    def test_build_model_multiple_calls_independence(self):
        """Test that multiple calls to build_model return independent model classes."""
        schema = InferredSchema(
            instructions="Test independence",
            examples_summary="Independence examples",
            examples_instructions_alignment="Independence ensures rebuilding yields fresh class objects",
            object_spec=ObjectSpec(
                name="IndependentRoot",
                fields=[
                    FieldSpec(name="test_field", type="string", description="Test field"),
                ],
            ),
            inference_prompt="Test prompt",
        )

        model_cls1 = schema.build_model()
        model_cls2 = schema.build_model()

        # Should be different class objects
        assert model_cls1 is not model_cls2

        # But should have the same structure
        assert model_cls1.model_fields.keys() == model_cls2.model_fields.keys()
        assert model_cls1.model_json_schema()["properties"] == model_cls2.model_json_schema()["properties"]

    def test_build_model_array_types(self):
        """Test that *_array types map to list element annotations and proper JSON Schema arrays."""
        schema = InferredSchema(
            instructions="Test array types",
            examples_summary="Array type examples",
            examples_instructions_alignment="Examples justify homogeneous primitive arrays",
            object_spec=ObjectSpec(
                name="ArrayRoot",
                fields=[
                    FieldSpec(name="tags_array", type="string_array", description="List of tag strings"),
                    FieldSpec(name="ids_array", type="integer_array", description="List of integer ids"),
                    FieldSpec(name="scores_array", type="float_array", description="List of float scores"),
                    FieldSpec(name="is_flags_array", type="boolean_array", description="List of boolean flags"),
                ],
            ),
            inference_prompt="Test prompt",
        )

        model_cls = schema.build_model()
        # Python annotations check (allow typing.List vs builtin list syntax)
        for field_name, inner in [
            ("tags_array", str),
            ("ids_array", int),
            ("scores_array", float),
            ("is_flags_array", bool),
        ]:
            ann = model_cls.model_fields[field_name].annotation
            assert get_origin(ann) is list, f"Origin for {field_name} should be list"
            assert get_args(ann) == (inner,), f"Inner type for {field_name} mismatch"

        js = model_cls.model_json_schema()
        props = js["properties"]
        assert props["tags_array"]["type"] == "array"
        assert props["tags_array"]["items"]["type"] == "string"
        assert props["ids_array"]["items"]["type"] == "integer"
        # Pydantic uses "number" for float
        assert props["scores_array"]["items"]["type"] == "number"
        assert props["is_flags_array"]["items"]["type"] == "boolean"
        # All required
        for name in ["tags_array", "ids_array", "scores_array", "is_flags_array"]:
            assert name in js["required"]
