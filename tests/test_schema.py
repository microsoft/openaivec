import os
import unittest
from unittest.mock import patch

from openai import OpenAI
from pydantic import BaseModel

from openaivec._schema import InferredSchema, SchemaInferenceInput, SchemaInferer  # type: ignore

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
        calls: list[int] = []

        def flaky_once(parsed):  # type: ignore
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("synthetic mismatch to trigger retry")
            return None

        with patch("openaivec._schema._basic_field_list_validation", side_effect=flaky_once):
            ds = self.DATASETS["retry_case"]
            client = OpenAI()
            inferer = SchemaInferer(client=client, model_name=SCHEMA_TEST_MODEL)
            suggestion = inferer.infer_schema(ds, max_retries=3)
        self.assertIsInstance(suggestion.fields, list)
        self.assertGreater(len(suggestion.fields), 0)
        for f in suggestion.fields:
            self.assertIn(f.type, {"string", "integer", "float", "boolean"})
            if f.enum_values is not None:
                self.assertEqual(f.type, "string")
                self.assertGreaterEqual(len(f.enum_values), 2)
                self.assertLessEqual(len(f.enum_values), 24)
        self.assertGreaterEqual(len(calls), 2)

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
