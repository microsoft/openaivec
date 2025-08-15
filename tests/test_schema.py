import unittest
from typing import List
from unittest.mock import patch

from openai import OpenAI
from pydantic import BaseModel

from openaivec._schema import (  # type: ignore
    ExampleData,
    SchemaSuggester,
    SuggestedSchema,
    _load_lenient_json,
    _validate_consistency,
    materialize_model,
)
from openaivec._serialize import deserialize_base_model


class TestSchemaSuggester(unittest.TestCase):
    @staticmethod
    def _assert_basic_invariants(t: "unittest.TestCase", s: SuggestedSchema) -> None:  # type: ignore[name-defined]
        t.assertIsInstance(s.purpose, str)
        t.assertTrue(s.purpose)
        t.assertIsInstance(s.example_data_description, str)
        t.assertTrue(s.example_data_description)
        t.assertIsInstance(s.suggested_schema, list)
        t.assertGreater(len(s.suggested_schema), 0)
        names = [f.name for f in s.suggested_schema]
        t.assertEqual(len(names), len(set(names)), "field names must be unique")
        for f in s.suggested_schema:
            t.assertIn(f.type, {"string", "integer", "float", "boolean"})
            t.assertTrue(f.description and isinstance(f.description, str))
            if f.enum_values is not None:
                t.assertEqual(f.type, "string")
                t.assertGreaterEqual(len(f.enum_values), 2)
                t.assertLessEqual(len(f.enum_values), 24)
        schema_dict = _load_lenient_json(s.json_schema_string)
        _validate_consistency(s, schema_dict)

    def test_basic(self):
        client = OpenAI()
        suggester = SchemaSuggester(client=client, model_name="gpt-5-mini")
        data = ExampleData(
            examples=[
                "Order #1234: customer requested refund due to damaged packaging.",
                "Order #1235: customer happy, praised fast shipping.",
                "Order #1236: delayed delivery complaint, wants status update.",
            ],
            purpose="Extract concise operational signals helpful for simple analytics (issue type, sentiment).",
        )
        suggestion = suggester.suggest_schema(data, max_retries=2)
        self._assert_basic_invariants(self, suggestion)

    def test_enum_detection(self):
        client = OpenAI()
        suggester = SchemaSuggester(client=client, model_name="gpt-5-mini")
        data = ExampleData(
            examples=[
                "Absolutely excellent product quality and fantastic support (clearly positive).",
                "Terrible experience and horrible customer support (clearly negative).",
                "Average / neutral overall, nothing especially good or bad (clearly neutral).",
                "Mixed feelings: mostly good build but disappointing battery (mixed).",
            ],
            purpose=(
                "Infer a categorical sentiment_label (small closed enum: e.g. positive, negative, neutral, mixed) "
                "and any other reliably extractable flat scalar signals. Provide enum values if evident."
            ),
        )
        suggestion = suggester.suggest_schema(data, max_retries=2)
        self._assert_basic_invariants(self, suggestion)
        has_enum = any(f.enum_values for f in suggestion.suggested_schema)
        if not has_enum:
            sentiment_like = [f for f in suggestion.suggested_schema if "sentiment" in f.name]
            self.assertTrue(sentiment_like, "expected a sentiment field or an enum with sentiment labels")

    def test_materialize_model(self):
        client = OpenAI()
        suggester = SchemaSuggester(client=client, model_name="gpt-5-mini")
        data = ExampleData(
            examples=[
                "Battery life is great and fast charging is helpful.",
                "Battery drains quickly and overheats sometimes.",
            ],
            purpose="Infer sentiment and one or two consistent quality signals as flat scalar fields.",
        )
        suggestion = suggester.suggest_schema(data, max_retries=2)
        model_cls = materialize_model(suggestion)
        self.assertTrue(issubclass(model_cls, BaseModel))
        schema = model_cls.model_json_schema()
        self.assertTrue("properties" in schema and schema["properties"], "expected non-empty properties")

    def test_retry(self):
        client = OpenAI()
        suggester = SchemaSuggester(client=client, model_name="gpt-5-mini")
        data = ExampleData(
            examples=[
                "User reported login failure after password reset.",
                "User confirmed issue was resolved after cache clear.",
            ],
            purpose="Extract key status signal (issue vs resolution) as minimal fields.",
        )

        calls: List[int] = []

        def flaky_once(parsed, schema_dict):  # type: ignore
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("synthetic mismatch to trigger retry")
            return None

        with patch("openaivec._schema._validate_consistency", side_effect=flaky_once):
            suggestion = suggester.suggest_schema(data, max_retries=3)
        self._assert_basic_invariants(self, suggestion)
        self.assertGreaterEqual(len(calls), 2, "should have retried after induced failure")

    def test_deserialize_base_model(self):
        client = OpenAI()
        suggester = SchemaSuggester(client=client, model_name="gpt-5-mini")
        data = ExampleData(
            examples=[
                "Great battery life and responsive UI.",
                "Battery drains fast and UI is sluggish.",
            ],
            purpose="Infer sentiment and one reliability signal as flat scalar fields.",
        )
        suggestion = suggester.suggest_schema(data, max_retries=2)
        schema_dict = _load_lenient_json(suggestion.json_schema_string)
        model_cls = deserialize_base_model(schema_dict)
        self.assertTrue(issubclass(model_cls, BaseModel))
        props = model_cls.model_json_schema().get("properties", {})
        self.assertEqual(list(props.keys()), [f.name for f in suggestion.suggested_schema])

    def test_feature_engineering_attrition(self):
        client = OpenAI()
        suggester = SchemaSuggester(client=client, model_name="gpt-5-mini")
        data = ExampleData(
            examples=[
                "Employee notes: workload high, considering options, manager support low.",
                "Employee notes: satisfied with team collaboration and growth opportunities.",
                "Employee notes: neutral performance period, seeking clearer career path.",
            ],
            purpose=(
                "Predict employee attrition probability; propose explanatory text-derived features only (no direct target)."
            ),
        )
        suggestion = suggester.suggest_schema(data, max_retries=2)
        self._assert_basic_invariants(self, suggestion)
        field_names = {f.name for f in suggestion.suggested_schema}
        forbidden = {"attrition_probability", "will_leave", "leave_label"}
        self.assertFalse(field_names & forbidden, f"Forbidden target-like fields present: {field_names & forbidden}")
        schema_dict = _load_lenient_json(suggestion.json_schema_string)
        model_cls = deserialize_base_model(schema_dict)
        self.assertTrue(issubclass(model_cls, BaseModel))

    def test_feature_engineering_purchase(self):
        client = OpenAI()
        suggester = SchemaSuggester(client=client, model_name="gpt-5-mini")
        data = ExampleData(
            examples=[
                "User viewed premium plan pricing and asked about discount.",
                "User compared basic vs pro, mentioned limited budget.",
                "User requested integration docs and asked about SLA terms.",
            ],
            purpose=(
                "Predict purchase probability; propose only explanatory scalar features (no purchase outcome field)."
            ),
        )
        suggestion = suggester.suggest_schema(data, max_retries=2)
        self._assert_basic_invariants(self, suggestion)
        field_names = {f.name for f in suggestion.suggested_schema}
        forbidden = {"purchase_probability", "will_buy", "purchase_label"}
        self.assertFalse(field_names & forbidden, f"Forbidden target-like fields present: {field_names & forbidden}")
        schema_dict = _load_lenient_json(suggestion.json_schema_string)
        model_cls = deserialize_base_model(schema_dict)
        self.assertTrue(issubclass(model_cls, BaseModel))

    def test_multiple_example_sets_deserialize(self):
        """Integration: multiple real prompts must yield JSON schema parsable into a dynamic model."""
        client = OpenAI()
        suggester = SchemaSuggester(client=client, model_name="gpt-5-mini")
        datasets: List[ExampleData] = [
            ExampleData(
                examples=[
                    "Order #9912 refunded due to damaged item.",
                    "Order #9913 resolved: replacement shipped.",
                    "Order #9914 pending investigation about missing parts.",
                ],
                purpose="Infer minimal operational status and high-level issue classification for analytics.",
            ),
            ExampleData(
                examples=[
                    "Rating: 4.5/5 - Fast delivery, packaging slightly dented.",
                    "Rating: 2/5 - Slow shipping and unresponsive support.",
                    "Rating: 5/5 - Excellent build quality and service.",
                ],
                purpose="Infer numeric rating and sentiment category as flat scalar features.",
            ),
            ExampleData(
                examples=[
                    "User: can't login after password reset though reset email succeeded.",
                    "User: login works but dashboard widgets fail to load intermittently.",
                    "User: account locked after multiple 2FA code failures.",
                ],
                purpose="Extract concise issue type classification and any consistently extractable signal.",
            ),
        ]

        for data in datasets:
            suggestion = suggester.suggest_schema(data, max_retries=3)
            self._assert_basic_invariants(self, suggestion)
            schema_dict = _load_lenient_json(suggestion.json_schema_string)
            model_cls = deserialize_base_model(schema_dict)
            self.assertTrue(issubclass(model_cls, BaseModel))
            props = model_cls.model_json_schema().get("properties", {})
            self.assertTrue(props, "expected non-empty properties")
            for spec in props.values():
                primitive_types = {"string", "integer", "number", "boolean"}
                t = spec.get("type")
                if t is None and "anyOf" in spec:
                    any_of = spec["anyOf"]
                    if isinstance(any_of, list) and 1 <= len(any_of) <= 2:
                        prims = [
                            x.get("type") for x in any_of if isinstance(x, dict) and x.get("type") in primitive_types
                        ]
                        nulls = [x for x in any_of if isinstance(x, dict) and x.get("type") == "null"]
                        self.assertEqual(len(prims), 1, "unexpected anyOf structure")
                        self.assertLessEqual(len(nulls), 1, "unexpected anyOf null structure")
                    else:
                        self.fail("unexpected anyOf structure for optional primitive")
                else:
                    self.assertIn(t, primitive_types)
                self.assertNotIn("items", spec)
                self.assertNotEqual(spec.get("type"), "object")
                if "enum" in spec:
                    self.assertTrue(isinstance(spec["enum"], list) and 2 <= len(spec["enum"]) <= 24)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
