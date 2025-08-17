import os
import unittest
from enum import Enum
from unittest.mock import patch

from openai import OpenAI
from pydantic import BaseModel

from openaivec._schema import FieldSpec, InferredSchema, SchemaInferenceInput, SchemaInferer  # type: ignore

SCHEMA_TEST_MODEL = "gpt-4.1-mini"


class TestSchemaInferer(unittest.TestCase):
    # Minimal datasets: one normal case + one for retry logic
    DATASETS: dict[str, SchemaInferenceInput] = {
        "basic_support": SchemaInferenceInput(
            examples=[
                "Order #1234: customer requested refund due to damaged packaging.",
                "Order #1235: customer happy, praised fast shipping.",
                "Order #1236: delayed delivery complaint, wants status update.",
            ],
            purpose="Extract useful flat analytic signals from short support notes.",
        ),
        "retry_case": SchemaInferenceInput(
            examples=[
                "User reported login failure after password reset.",
                "User confirmed issue was resolved after cache clear.",
            ],
            purpose="Infer minimal status/phase signals from event style notes.",
        ),
    }

    INFERRED: dict[str, InferredSchema] = {}

    @classmethod
    def setUpClass(cls):  # noqa: D401 - standard unittest hook
        """Infer schemas for all datasets once (live API) to reuse across tests."""
        if "OPENAI_API_KEY" not in os.environ:
            raise RuntimeError("OPENAI_API_KEY not set (tests require real API per project policy)")
        client = OpenAI()
        inferer = SchemaInferer(client=client, model_name=SCHEMA_TEST_MODEL)
        for name, ds in cls.DATASETS.items():
            cls.INFERRED[name] = inferer.infer_schema(ds, max_retries=2)

    def test_inference_basic(self):
        for inferred in self.INFERRED.values():
            self.assertIsInstance(inferred.fields, list)
            self.assertGreater(len(inferred.fields), 0)
            for f in inferred.fields:
                self.assertIn(f.type, {"string", "integer", "float", "boolean"})
                if f.enum_values is not None:
                    self.assertEqual(f.type, "string")
                    self.assertGreaterEqual(len(f.enum_values), 2)
                    self.assertLessEqual(len(f.enum_values), 24)

    def test_build_model(self):
        inferred = self.INFERRED["basic_support"]
        model_cls = inferred.build_model()
        self.assertTrue(issubclass(model_cls, BaseModel))
        props = model_cls.model_json_schema().get("properties", {})
        self.assertTrue(props)

    def test_retry(self):
        call_count = 0

        def fail_first_call(parsed):  # type: ignore
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("synthetic mismatch to trigger retry")
            # Let subsequent calls succeed by not raising

        with patch("openaivec._schema._basic_field_list_validation", side_effect=fail_first_call):
            ds = self.DATASETS["retry_case"]
            client = OpenAI()
            inferer = SchemaInferer(client=client, model_name=SCHEMA_TEST_MODEL)
            suggestion = inferer.infer_schema(ds, max_retries=3)

        # Verify the suggestion is valid
        self.assertIsInstance(suggestion.fields, list)
        self.assertGreater(len(suggestion.fields), 0)
        for f in suggestion.fields:
            self.assertIn(f.type, {"string", "integer", "float", "boolean"})
            if f.enum_values is not None:
                self.assertEqual(f.type, "string")
                self.assertGreaterEqual(len(f.enum_values), 2)
                self.assertLessEqual(len(f.enum_values), 24)

        # Verify that retry mechanism was triggered - should have at least 2 calls
        # (first fails, second succeeds)
        self.assertGreaterEqual(call_count, 2, f"Expected at least 2 validation calls, got {call_count}")

    def test_structuring_basic(self):
        inferred = self.INFERRED["basic_support"]
        raw = self.DATASETS["basic_support"].examples[0]
        client = OpenAI()
        model_cls = inferred.build_model()
        parsed = client.responses.parse(
            model=SCHEMA_TEST_MODEL,
            instructions=inferred.inference_prompt,
            input=raw,
            text_format=model_cls,
        )
        structured = parsed.output_parsed
        self.assertIsInstance(structured, BaseModel)

    def test_field_descriptions_in_model(self):
        """Test that field descriptions from FieldSpec are reflected in generated Pydantic model."""
        inferred = self.INFERRED["basic_support"]
        model_cls = inferred.build_model()

        # Get the model schema which includes field descriptions
        schema = model_cls.model_json_schema()
        properties = schema.get("properties", {})

        # Verify that all fields from the inferred schema have descriptions in the model
        for field_spec in inferred.fields:
            field_name = field_spec.name
            self.assertIn(field_name, properties, f"Field '{field_name}' should be in model properties")

            field_schema = properties[field_name]
            self.assertIn("description", field_schema, f"Field '{field_name}' should have a description")
            self.assertEqual(
                field_schema["description"],
                field_spec.description,
                f"Field '{field_name}' description should match FieldSpec description",
            )


class TestInferredSchemaBuildModel(unittest.TestCase):
    """Comprehensive MECE test cases for InferredSchema.build_model method."""

    def test_build_model_primitive_types(self):
        """Test that all primitive types are correctly mapped to Python types."""
        schema = InferredSchema(
            purpose="Test primitive types",
            examples_summary="Various primitive type examples",
            fields=[
                FieldSpec(name="text_field", type="string", description="A string field"),
                FieldSpec(name="number_field", type="integer", description="An integer field"),
                FieldSpec(name="decimal_field", type="float", description="A float field"),
                FieldSpec(name="flag_field", type="boolean", description="A boolean field"),
            ],
            inference_prompt="Test prompt",
        )

        model_cls = schema.build_model()
        schema_dict = model_cls.model_json_schema()
        properties = schema_dict["properties"]

        # Verify correct type mapping
        self.assertEqual(properties["text_field"]["type"], "string")
        self.assertEqual(properties["number_field"]["type"], "integer")
        self.assertEqual(properties["decimal_field"]["type"], "number")
        self.assertEqual(properties["flag_field"]["type"], "boolean")

        # Verify all fields are required
        self.assertEqual(set(schema_dict["required"]), {"text_field", "number_field", "decimal_field", "flag_field"})

    def test_build_model_enum_field(self):
        """Test that enum fields generate proper Enum classes."""
        schema = InferredSchema(
            purpose="Test enum types",
            examples_summary="Enum examples",
            fields=[
                FieldSpec(
                    name="status_field",
                    type="string",
                    description="Status enum field",
                    enum_values=["active", "inactive", "pending"],
                ),
                FieldSpec(name="regular_field", type="string", description="Regular string field"),
            ],
            inference_prompt="Test prompt",
        )

        model_cls = schema.build_model()

        # Verify enum field type
        status_annotation = model_cls.model_fields["status_field"].annotation
        self.assertTrue(issubclass(status_annotation, Enum))

        # Verify enum values
        enum_values = [member.value for member in status_annotation]
        self.assertEqual(set(enum_values), {"active", "inactive", "pending"})

        # Verify non-enum field is still string
        regular_annotation = model_cls.model_fields["regular_field"].annotation
        self.assertEqual(regular_annotation, str)

    def test_build_model_enum_name_sanitization(self):
        """Test that enum member names are properly sanitized."""
        schema = InferredSchema(
            purpose="Test enum sanitization",
            examples_summary="Enum sanitization examples",
            fields=[
                FieldSpec(
                    name="complex_status",
                    type="string",
                    description="Complex enum with special characters",
                    enum_values=["high-priority", "low priority", "2nd-level", "normal"],
                ),
            ],
            inference_prompt="Test prompt",
        )

        model_cls = schema.build_model()
        enum_cls = model_cls.model_fields["complex_status"].annotation

        # Verify sanitized member names exist
        member_names = [member.name for member in enum_cls]
        self.assertIn("HIGH_PRIORITY", member_names)
        self.assertIn("LOW_PRIORITY", member_names)
        self.assertIn("V_2ND_LEVEL", member_names)  # Should be prefixed with V_ for digit start
        self.assertIn("NORMAL", member_names)

    def test_build_model_enum_collision_handling(self):
        """Test that enum member name collisions are handled properly."""
        schema = InferredSchema(
            purpose="Test enum collision handling",
            examples_summary="Enum collision examples",
            fields=[
                FieldSpec(
                    name="collision_test",
                    type="string",
                    description="Enum with potential collisions",
                    enum_values=["test", "test-1", "test_1"],  # These could all map to TEST_1
                ),
            ],
            inference_prompt="Test prompt",
        )

        model_cls = schema.build_model()
        enum_cls = model_cls.model_fields["collision_test"].annotation

        # Verify all values are preserved with unique member names
        member_names = [member.name for member in enum_cls]
        self.assertEqual(len(member_names), 3)  # Should have 3 unique member names

        # Verify original values are preserved
        member_values = [member.value for member in enum_cls]
        self.assertEqual(set(member_values), {"test", "test-1", "test_1"})

    def test_build_model_field_ordering(self):
        """Test that field ordering is preserved in the generated model."""
        fields = [
            FieldSpec(name="third_field", type="string", description="Third field"),
            FieldSpec(name="first_field", type="integer", description="First field"),
            FieldSpec(name="second_field", type="boolean", description="Second field"),
        ]

        schema = InferredSchema(
            purpose="Test field ordering",
            examples_summary="Field ordering examples",
            fields=fields,
            inference_prompt="Test prompt",
        )

        model_cls = schema.build_model()
        model_field_names = list(model_cls.model_fields.keys())
        expected_order = ["third_field", "first_field", "second_field"]

        self.assertEqual(model_field_names, expected_order)

    def test_build_model_field_descriptions(self):
        """Test that field descriptions are correctly included in the model."""
        schema = InferredSchema(
            purpose="Test field descriptions",
            examples_summary="Description examples",
            fields=[
                FieldSpec(name="described_field", type="string", description="This is a detailed description"),
                FieldSpec(name="another_field", type="integer", description="Another detailed description"),
            ],
            inference_prompt="Test prompt",
        )

        model_cls = schema.build_model()
        schema_dict = model_cls.model_json_schema()
        properties = schema_dict["properties"]

        self.assertEqual(properties["described_field"]["description"], "This is a detailed description")
        self.assertEqual(properties["another_field"]["description"], "Another detailed description")

    def test_build_model_empty_fields(self):
        """Test behavior with empty fields list."""
        schema = InferredSchema(
            purpose="Test empty fields", examples_summary="Empty examples", fields=[], inference_prompt="Test prompt"
        )

        model_cls = schema.build_model()
        self.assertEqual(len(model_cls.model_fields), 0)

        # Should still be a valid BaseModel
        self.assertTrue(issubclass(model_cls, BaseModel))

        # Should be able to instantiate with no arguments
        instance = model_cls()
        self.assertIsInstance(instance, BaseModel)

    def test_build_model_mixed_enum_and_regular_fields(self):
        """Test a complex scenario with both enum and regular fields of all types."""
        schema = InferredSchema(
            purpose="Test mixed field types",
            examples_summary="Mixed type examples",
            fields=[
                FieldSpec(
                    name="priority", type="string", description="Priority level", enum_values=["high", "medium", "low"]
                ),
                FieldSpec(name="count", type="integer", description="Item count"),
                FieldSpec(name="score", type="float", description="Quality score"),
                FieldSpec(name="is_active", type="boolean", description="Active status"),
                FieldSpec(name="category", type="string", description="Category name", enum_values=["A", "B", "C"]),
                FieldSpec(name="description", type="string", description="Free text description"),
            ],
            inference_prompt="Test prompt",
        )

        model_cls = schema.build_model()

        # Verify enum fields
        priority_type = model_cls.model_fields["priority"].annotation
        category_type = model_cls.model_fields["category"].annotation
        self.assertTrue(issubclass(priority_type, Enum))
        self.assertTrue(issubclass(category_type, Enum))

        # Verify regular fields
        self.assertEqual(model_cls.model_fields["count"].annotation, int)
        self.assertEqual(model_cls.model_fields["score"].annotation, float)
        self.assertEqual(model_cls.model_fields["is_active"].annotation, bool)
        self.assertEqual(model_cls.model_fields["description"].annotation, str)

        # Verify all fields are present
        self.assertEqual(len(model_cls.model_fields), 6)

    def test_build_model_multiple_calls_independence(self):
        """Test that multiple calls to build_model return independent model classes."""
        schema = InferredSchema(
            purpose="Test independence",
            examples_summary="Independence examples",
            fields=[
                FieldSpec(name="test_field", type="string", description="Test field"),
            ],
            inference_prompt="Test prompt",
        )

        model_cls1 = schema.build_model()
        model_cls2 = schema.build_model()

        # Should be different class objects
        self.assertIsNot(model_cls1, model_cls2)

        # But should have the same structure
        self.assertEqual(model_cls1.model_fields.keys(), model_cls2.model_fields.keys())
        self.assertEqual(model_cls1.model_json_schema()["properties"], model_cls2.model_json_schema()["properties"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
