import logging
import unittest

from openai import OpenAI
from pydantic import BaseModel

from openaivec.prompt import FewShotPromptBuilder

logging.basicConfig(level=logging.INFO, force=True)


class TestAtomicPromptBuilder(unittest.TestCase):
    def setUp(self):
        self.client: OpenAI = OpenAI()
        self.model_name: str = "gpt-4.1-nano"

    def test_improve(self):
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

    def test_improve_ja(self):
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
