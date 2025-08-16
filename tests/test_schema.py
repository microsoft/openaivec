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

    @staticmethod
    def _assert_basic_invariants(t: "unittest.TestCase", s: InferredSchema) -> None:  # type: ignore[name-defined]
        t.assertIsInstance(s.fields, list)
        t.assertGreater(len(s.fields), 0)
        for f in s.fields:
            t.assertIn(f.type, {"string", "integer", "float", "boolean"})
            if f.enum_values is not None:
                t.assertEqual(f.type, "string")
                t.assertGreaterEqual(len(f.enum_values), 2)
                t.assertLessEqual(len(f.enum_values), 24)

    def _infer(self, examples: list[str], purpose: str, retries: int = 2) -> InferredSchema:
        if "OPENAI_API_KEY" not in os.environ:
            self.fail("OPENAI_API_KEY not set (tests require real API per project policy)")
        client = OpenAI()
        inferer = SchemaInferer(client=client, model_name=SCHEMA_TEST_MODEL)
        data = SchemaInferenceInput(examples=examples, purpose=purpose)
        return inferer.infer_schema(data, max_retries=retries)

    def test_inference_basic(self):
        for inferred in self.INFERRED.values():
            self._assert_basic_invariants(self, inferred)

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
            suggestion = self._infer(ds.examples, ds.purpose, retries=3)
        self._assert_basic_invariants(self, suggestion)
        self.assertGreaterEqual(len(calls), 2)

    def test_structuring_basic(self):
        inferred = self.INFERRED["basic_support"]
        raw = self.DATASETS["basic_support"].examples[0]
        structured = self._structure_one_example(inferred, raw)
        self.assertIsInstance(structured, BaseModel)

    def _structure_one_example(
        self, suggestion: InferredSchema, example_text: str, model_name: str = SCHEMA_TEST_MODEL
    ):
        """Perform a real second pass structuring call using the schema's inference_prompt.

        Args:
            suggestion (InferredSchema): Previously inferred schema via Responses.parse.
            example_text (str): One raw example text (must be from the original examples set per contract).
            model_name (str): Model/deployment name.
        Returns:
            BaseModel: Parsed structured instance produced by second pass.
        """
        client = OpenAI()
        model_cls = suggestion.build_model()
        parsed = client.responses.parse(
            model=model_name,
            instructions=suggestion.inference_prompt,
            input=example_text,
            text_format=model_cls,
        )
        return parsed.output_parsed

    # Removed all other structuring variants; they provided only name-sensitive coverage.


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
