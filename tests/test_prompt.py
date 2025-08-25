import io
import logging
import sys

import pytest
from openai import OpenAI
from pydantic import BaseModel

from openaivec._prompt import FewShotPromptBuilder

logging.basicConfig(level=logging.INFO, force=True)


class TestAtomicPromptBuilder:
    @pytest.fixture
    def setup(self):
        """Setup for tests."""
        self.client = OpenAI()
        self.model_name = "gpt-4.1-nano"

    def test_improve(self, setup):
        prompt: str = (
            FewShotPromptBuilder()
            .purpose("Return the smallest category that includes the given word")
            .caution("Never use proper nouns as categories")
            .example("Apple", "Fruit")
            .example("Car", "Vehicle")
            .example("Tokyo", "City")
            .example("Keiichi Sogabe", "Musician")
            .example("America", "Country")
            .example("United Kingdom", "Country")
            # Examples of countries
            .example("France", "Country")
            .example("Germany", "Country")
            .example("Brazil", "Country")
            # Examples of famous Americans
            .example("Elvis Presley", "Musician")
            .example("Marilyn Monroe", "Actor")
            .example("Michael Jordan", "Athlete")
            # Examples of American place names
            .example("New York", "City")
            .example("Los Angeles", "City")
            .example("Grand Canyon", "Natural Landmark")
            # Examples of everyday items
            .example("Toothbrush", "Hygiene Product")
            .example("Notebook", "Stationery")
            .example("Spoon", "Kitchenware")
            # Examples of company names
            .example("Google", "Company in USA")
            .example("Toyota", "Company in Japan")
            .example("Amazon", "Company in USA")
            # Examples of abstract concepts
            .example("Freedom", "Abstract Idea")
            .example("Happiness", "Emotion")
            .example("Justice", "Ethical Principle")
            # Steve Wozniak is not boring
            .example("Steve Wozniak", "is not boring")
            .improve(self.client, self.model_name)
            .explain()
            .build()
        )

        # Log the parsed XML result
        logging.info("Parsed XML: %s", prompt)

    def test_improve_ja(self, setup):
        prompt: str = (
            FewShotPromptBuilder()
            .purpose("受け取った単語を含む最小のカテゴリ名を返してください。")
            .caution("カテゴリ名に固有名詞を使用しないでください")
            .caution("単語としてはWikipediaに載るような、あらゆる単語が想定されるので注意が必要です。")
            .example("りんご", "果物")
            .example("パンダ", "クマ科")
            .example("東京", "都市")
            .example("ネコ", "ネコ科")
            .example("アメリカ", "国")
            .improve(self.client, self.model_name)
            .explain()
            .build()
        )

        logging.info("Prompt: %s", prompt)

    def test_with_basemodel(self):
        class Fruit(BaseModel):
            name: str
            color: str

        prompt: str = (
            FewShotPromptBuilder()
            .purpose("Return the smallest category that includes the given word")
            .caution("Never use proper nouns as categories")
            .example("Apple", Fruit(name="Apple", color="Red"))
            .example("Peach", Fruit(name="Peach", color="Pink"))
            .example("Banana", Fruit(name="Banana", color="Yellow"))
            .build()
        )

        logging.info(prompt)

    def test_improve_without_args(self):
        """Test improve method using DI container (no explicit client/model_name)."""
        prompt: str = (
            FewShotPromptBuilder()
            .purpose("Classify the given text by sentiment")
            .caution("Consider context and nuance")
            .example("I love this!", "positive")
            .example("This is terrible", "negative")
            .example("It's okay", "neutral")
            .improve()  # Uses DI container with environment variables
            .build()
        )

        # Should complete without error if OPENAI_API_KEY is set
        assert prompt is not None
        logging.info("Improved prompt (DI): %s", prompt)

    def test_explain_without_improve(self):
        """Test explain method called without prior improve() call."""
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        builder = (
            FewShotPromptBuilder()
            .purpose("Test purpose")
            .example("input", "output")
            .explain()  # Should handle gracefully
        )

        # Restore stdout
        sys.stdout = sys.__stdout__

        # Check that appropriate message was printed
        output = captured_output.getvalue()
        assert "No improvement steps available" in output

        # Builder should still be usable
        prompt = builder.build()
        assert prompt is not None
