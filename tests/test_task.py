from dataclasses import FrozenInstanceError

import pytest
from pydantic import BaseModel, Field, ValidationError

from openaivec._model import PreparedTask
from openaivec.task.nlp import multilingual_translation
from openaivec.task.nlp.translation import TranslatedString


class SimpleResponse(BaseModel):
    message: str = Field(description="A simple message")
    count: int = Field(description="A count value", default=1)


class TestPreparedTask:
    def test_prepared_task_creation(self):
        task = PreparedTask(instructions="Test instruction", response_format=SimpleResponse)

        assert task.instructions == "Test instruction"
        assert task.response_format == SimpleResponse
        with pytest.raises(AttributeError):
            _ = getattr(task, "default_api_kwargs")

    def test_prepared_task_is_frozen(self):
        task = PreparedTask(instructions="Test instruction", response_format=SimpleResponse)

        with pytest.raises(FrozenInstanceError):
            setattr(task, "instructions", "Modified instruction")

    def test_prepared_task_response_format_type(self):
        task = PreparedTask(instructions="Test", response_format=SimpleResponse)

        assert isinstance(task.response_format, type)
        assert issubclass(task.response_format, BaseModel)


class TestMultilingualTranslationTask:
    def test_multilingual_translation_task_exists(self):
        assert isinstance(multilingual_translation(), PreparedTask)

    def test_multilingual_translation_task_configuration(self):
        task = multilingual_translation()

        assert task.instructions
        assert task.response_format == TranslatedString

    def test_multilingual_translation_task_instructions(self):
        instructions = multilingual_translation().instructions
        assert "translate" in instructions.lower()

    def test_translated_string_model_fields(self):
        sample_data = {
            "en": "Hello",
            "de": "Hallo",
            "nl": "Hallo",
            "sv": "Hej",
            "da": "Hej",
            "no": "Hei",
            "es": "Hola",
            "fr": "Bonjour",
            "it": "Ciao",
            "pt": "Ola",
            "ro": "Salut",
            "ca": "Hola",
            "ru": "Privet",
            "pl": "Czesc",
            "cs": "Ahoj",
            "sk": "Ahoj",
            "uk": "Pryvit",
            "bg": "Zdravey",
            "hr": "Bok",
            "sr": "Zdravo",
            "ja": "konnichiwa",
            "ko": "annyeong",
            "zh": "nihao",
            "zh_tw": "nihao",
            "hi": "namaste",
            "bn": "hello",
            "te": "hello",
            "ta": "hello",
            "ur": "salam",
            "th": "hello",
            "vi": "xin chao",
            "id": "halo",
            "ms": "hello",
            "tl": "kumusta",
            "ar": "marhaba",
            "he": "shalom",
            "fa": "salam",
            "tr": "merhaba",
            "sw": "hujambo",
            "am": "selam",
            "fi": "hei",
            "hu": "szia",
            "et": "tere",
            "lv": "sveiki",
            "lt": "labas",
            "el": "geia",
            "is_": "hallo",
            "eu": "kaixo",
            "cy": "helo",
            "ga": "dia dhuit",
            "mt": "bonu",
        }

        translated_string = TranslatedString(**sample_data)

        assert translated_string.en == "Hello"
        assert translated_string.ja == "konnichiwa"
        assert translated_string.es == "Hola"
        assert translated_string.zh == "nihao"

    def test_translated_string_required_fields(self):
        with pytest.raises(ValidationError):
            TranslatedString.model_validate({"en": "Hello", "ja": "konnichiwa"})

    def test_translated_string_field_types(self):
        fields = TranslatedString.model_fields

        assert "en" in fields
        assert "ja" in fields
        assert "es" in fields
        assert "zh" in fields

        for field_info in fields.values():
            assert field_info.annotation is str
