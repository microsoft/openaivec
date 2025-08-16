import os
import unittest
from unittest.mock import patch

from openai import OpenAI
from pydantic import BaseModel

from openaivec._schema import InferredSchema, SchemaInferenceInput, SchemaInferer  # type: ignore

# Use a non-reasoning mainstream model for all schema tests
SCHEMA_TEST_MODEL = "gpt-4.1-mini"


class TestSchemaInferer(unittest.TestCase):
    # Central catalog of schema inference scenarios
    DATASETS: dict[str, SchemaInferenceInput] = {
        "operational_status": SchemaInferenceInput(
            examples=[
                "Order #9912 refunded due to damaged item.",
                "Order #9913 resolved: replacement shipped.",
                "Order #9914 pending investigation about missing parts.",
            ],
            purpose="Infer minimal operational status and high-level issue classification for analytics.",
        ),
        "rating_sentiment": SchemaInferenceInput(
            examples=[
                "Rating: 4.5/5 - Fast delivery, packaging slightly dented.",
                "Rating: 2/5 - Slow shipping and unresponsive support.",
                "Rating: 5/5 - Excellent build quality and service.",
            ],
            purpose="Infer numeric rating and sentiment category as flat scalar features.",
        ),
        "issue_type_login": SchemaInferenceInput(
            examples=[
                "User: can't login after password reset though reset email succeeded.",
                "User: login works but dashboard widgets fail to load intermittently.",
                "User: account locked after multiple 2FA code failures.",
            ],
            purpose="Extract concise issue type classification and any consistently extractable signal.",
        ),
        "basic_support_signals": SchemaInferenceInput(
            examples=[
                "Order #1234: customer requested refund due to damaged packaging.",
                "Order #1235: customer happy, praised fast shipping.",
                "Order #1236: delayed delivery complaint, wants status update.",
            ],
            purpose="Extract concise operational signals helpful for simple analytics (issue type, sentiment).",
        ),
        "sentiment_enum_detection": SchemaInferenceInput(
            examples=[
                "Absolutely excellent product quality and fantastic support (clearly positive).",
                "Terrible experience and horrible customer support (clearly negative).",
                "Average / neutral overall, nothing especially good or bad (clearly neutral).",
                "Mixed feelings: mostly good build but disappointing battery (mixed).",
            ],
            purpose=(
                "Infer a categorical sentiment_label (small closed enum: e.g. positive, negative, neutral, mixed) and any other reliably extractable flat scalar signals. Provide enum values if evident."
            ),
        ),
        "battery_quality": SchemaInferenceInput(
            examples=[
                "Battery life is great and fast charging is helpful.",
                "Battery drains quickly and overheats sometimes.",
            ],
            purpose="Infer sentiment and one or two consistent quality signals as flat scalar fields.",
        ),
        "shipping_quality_simple": SchemaInferenceInput(
            examples=[
                "Positive experience: fast shipping and solid build quality.",
                "Negative: slow delivery and fragile packaging.",
            ],
            purpose="Infer a sentiment_label enum (positive/negative) plus any one additional quality signal if stable.",
        ),
        "status_resolution_retry": SchemaInferenceInput(
            examples=[
                "User reported login failure after password reset.",
                "User confirmed issue was resolved after cache clear.",
            ],
            purpose="Extract key status signal (issue vs resolution) as minimal fields.",
        ),
        "battery_ui_reliability": SchemaInferenceInput(
            examples=[
                "Great battery life and responsive UI.",
                "Battery drains fast and UI is sluggish.",
            ],
            purpose="Infer sentiment and one reliability signal as flat scalar fields.",
        ),
        "attrition_features": SchemaInferenceInput(
            examples=[
                "Employee notes: workload high, considering options, manager support low.",
                "Employee notes: satisfied with team collaboration and growth opportunities.",
                "Employee notes: neutral performance period, seeking clearer career path.",
            ],
            purpose="Predict employee attrition probability; propose explanatory text-derived features only (no direct target).",
        ),
        "purchase_features": SchemaInferenceInput(
            examples=[
                "User viewed premium plan pricing and asked about discount.",
                "User compared basic vs pro, mentioned limited budget.",
                "User requested integration docs and asked about SLA terms.",
            ],
            purpose="Predict purchase probability; propose only explanatory scalar features (no purchase outcome field).",
        ),
    }

    # Cache of inferred schemas (populated once in setUpClass to avoid duplicate API calls)
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
        t.assertIsInstance(s.purpose, str)
        t.assertTrue(s.purpose)
        t.assertIsInstance(s.examples_summary, str)
        t.assertTrue(s.examples_summary)
        t.assertIsInstance(s.fields, list)
        t.assertGreater(len(s.fields), 0)
        names = [f.name for f in s.fields]
        t.assertEqual(len(names), len(set(names)), "field names must be unique")
        for f in s.fields:
            t.assertIn(f.type, {"string", "integer", "float", "boolean"})
            t.assertTrue(f.description and isinstance(f.description, str))
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

    # --- A. Inference invariants across all datasets ---
    def test_inference_invariants_all_datasets(self):
        for name, inferred in self.INFERRED.items():
            with self.subTest(dataset=name):
                self._assert_basic_invariants(self, inferred)

    # --- B. Enum / categorical detection ---
    def test_enum_detection_sentiment_dataset(self):
        inferred = self.INFERRED["sentiment_enum_detection"]
        self._assert_basic_invariants(self, inferred)
        has_enum = any(f.enum_values for f in inferred.fields)
        if not has_enum:
            sentiment_like = [f for f in inferred.fields if "sentiment" in f.name]
            string_fields = [f for f in inferred.fields if f.type == "string"]
            self.assertTrue(sentiment_like or string_fields)

    # --- C. Model materialization (build_model) ---
    def test_build_model_all_datasets(self):
        for name, inferred in self.INFERRED.items():
            with self.subTest(dataset=name):
                model_cls = inferred.build_model()
                self.assertTrue(issubclass(model_cls, BaseModel))
                props = model_cls.model_json_schema().get("properties", {})
                self.assertTrue(props)
                self.assertEqual(set(props.keys()), {f.name for f in inferred.fields})

    def test_retry(self):
        calls: list[int] = []

        def flaky_once(parsed):  # type: ignore
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("synthetic mismatch to trigger retry")
            return None

        with patch("openaivec._schema._basic_field_list_validation", side_effect=flaky_once):
            ds = self.DATASETS["status_resolution_retry"]
            suggestion = self._infer(ds.examples, ds.purpose, retries=3)
        self._assert_basic_invariants(self, suggestion)
        self.assertGreaterEqual(len(calls), 2)

    # --- D. Feature engineering constraints (no target leakage) ---
    def test_feature_engineering_attrition_no_target(self):
        inferred = self.INFERRED["attrition_features"]
        field_names = {f.name for f in inferred.fields}
        forbidden = {"attrition_probability", "will_leave", "leave_label"}
        self.assertFalse(field_names & forbidden)
        inferred.build_model()

    def test_feature_engineering_purchase_no_target(self):
        inferred = self.INFERRED["purchase_features"]
        field_names = {f.name for f in inferred.fields}
        forbidden = {"purchase_probability", "will_buy", "purchase_label"}
        self.assertFalse(field_names & forbidden)
        inferred.build_model()

    # --- E. Second-pass structuring (Responses.parse with dynamic model) ---
    def test_structuring_operational_status(self):
        inferred = self.INFERRED["operational_status"]
        raw = self.DATASETS["operational_status"].examples[0]
        structured = self._structure_one_example(inferred, raw)
        self.assertIsInstance(structured, BaseModel)
        for f in inferred.fields:
            self.assertTrue(hasattr(structured, f.name))

    def test_structuring_rating_sentiment(self):
        inferred = self.INFERRED["rating_sentiment"]
        raw = self.DATASETS["rating_sentiment"].examples[0]
        structured = self._structure_one_example(inferred, raw)
        self.assertIsInstance(structured, BaseModel)
        names = {f.name for f in inferred.fields}
        self.assertTrue(any(k in names for k in ("rating", "sentiment", "sentiment_label")))
        for f in inferred.fields:
            self.assertTrue(hasattr(structured, f.name))

    def test_structuring_issue_type(self):
        inferred = self.INFERRED["issue_type_login"]
        raw = self.DATASETS["issue_type_login"].examples[0]
        structured = self._structure_one_example(inferred, raw)
        self.assertIsInstance(structured, BaseModel)
        for f in inferred.fields:
            self.assertTrue(hasattr(structured, f.name))

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

    def test_structuring_basic_support(self):
        inferred = self.INFERRED["basic_support_signals"]
        raw = self.DATASETS["basic_support_signals"].examples[0]
        structured = self._structure_one_example(inferred, raw)
        self.assertIsInstance(structured, BaseModel)
        for f in inferred.fields:
            self.assertTrue(hasattr(structured, f.name))

    def test_structuring_attrition_features(self):
        inferred = self.INFERRED["attrition_features"]
        raw = self.DATASETS["attrition_features"].examples[0]
        structured = self._structure_one_example(inferred, raw)
        self.assertIsInstance(structured, BaseModel)
        forbidden = {"attrition_probability", "will_leave", "leave_label"}
        for name in forbidden:
            self.assertFalse(hasattr(structured, name))

    def test_structuring_purchase_features(self):
        inferred = self.INFERRED["purchase_features"]
        raw = self.DATASETS["purchase_features"].examples[0]
        structured = self._structure_one_example(inferred, raw)
        self.assertIsInstance(structured, BaseModel)
        forbidden = {"purchase_probability", "will_buy", "purchase_label"}
        for name in forbidden:
            self.assertFalse(hasattr(structured, name))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
