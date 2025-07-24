import unittest
from dataclasses import FrozenInstanceError

from pydantic import BaseModel, Field

from openaivec.task import PreparedTask
from openaivec.task.nlp import MULTILINGUAL_TRANSLATION
from openaivec.task.nlp.translation import TranslatedString


class SimpleResponse(BaseModel):
    """Simple response model for testing."""
    message: str = Field(description="A simple message")
    count: int = Field(description="A count value", default=1)


class TestPreparedTask(unittest.TestCase):
    """Unit tests for PreparedTask class."""

    def test_prepared_task_creation_with_defaults(self):
        """Test creating a PreparedTask with default parameters."""
        task = PreparedTask(
            instructions="Test instruction",
            response_format=SimpleResponse
        )
        
        self.assertEqual(task.instructions, "Test instruction")
        self.assertEqual(task.response_format, SimpleResponse)
        self.assertEqual(task.temperature, 0.0)
        self.assertEqual(task.top_p, 1.0)

    def test_prepared_task_creation_with_custom_parameters(self):
        """Test creating a PreparedTask with custom parameters."""
        task = PreparedTask(
            instructions="Custom instruction",
            response_format=SimpleResponse,
            temperature=0.7,
            top_p=0.9
        )
        
        self.assertEqual(task.instructions, "Custom instruction")
        self.assertEqual(task.response_format, SimpleResponse)
        self.assertEqual(task.temperature, 0.7)
        self.assertEqual(task.top_p, 0.9)

    def test_prepared_task_is_frozen(self):
        """Test that PreparedTask is immutable (frozen)."""
        task = PreparedTask(
            instructions="Test instruction",
            response_format=SimpleResponse
        )
        
        # Attempting to modify any field should raise FrozenInstanceError
        with self.assertRaises(FrozenInstanceError):
            task.instructions = "Modified instruction"
        
        with self.assertRaises(FrozenInstanceError):
            task.temperature = 0.5

    def test_prepared_task_temperature_bounds(self):
        """Test PreparedTask accepts valid temperature values."""
        # Test minimum temperature
        task_min = PreparedTask(
            instructions="Test",
            response_format=SimpleResponse,
            temperature=0.0
        )
        self.assertEqual(task_min.temperature, 0.0)
        
        # Test maximum temperature
        task_max = PreparedTask(
            instructions="Test",
            response_format=SimpleResponse,
            temperature=1.0
        )
        self.assertEqual(task_max.temperature, 1.0)
        
        # Test intermediate value
        task_mid = PreparedTask(
            instructions="Test",
            response_format=SimpleResponse,
            temperature=0.5
        )
        self.assertEqual(task_mid.temperature, 0.5)

    def test_prepared_task_top_p_bounds(self):
        """Test PreparedTask accepts valid top_p values."""
        # Test minimum top_p
        task_min = PreparedTask(
            instructions="Test",
            response_format=SimpleResponse,
            top_p=0.0
        )
        self.assertEqual(task_min.top_p, 0.0)
        
        # Test maximum top_p
        task_max = PreparedTask(
            instructions="Test",
            response_format=SimpleResponse,
            top_p=1.0
        )
        self.assertEqual(task_max.top_p, 1.0)
        
        # Test intermediate value
        task_mid = PreparedTask(
            instructions="Test",
            response_format=SimpleResponse,
            top_p=0.5
        )
        self.assertEqual(task_mid.top_p, 0.5)

    def test_prepared_task_response_format_type(self):
        """Test that response_format must be a Pydantic BaseModel type."""
        task = PreparedTask(
            instructions="Test",
            response_format=SimpleResponse
        )
        
        # Verify the response_format is a type, not an instance
        self.assertTrue(isinstance(task.response_format, type))
        self.assertTrue(issubclass(task.response_format, BaseModel))


class TestMultilingualTranslationTask(unittest.TestCase):
    """Unit tests for MULTILINGUAL_TRANSLATION."""

    def test_multilingual_translation_task_exists(self):
        """Test that MULTILINGUAL_TRANSLATION is properly defined."""
        self.assertIsInstance(MULTILINGUAL_TRANSLATION, PreparedTask)

    def test_multilingual_translation_task_configuration(self):
        """Test the configuration of MULTILINGUAL_TRANSLATION."""
        task = MULTILINGUAL_TRANSLATION
        
        # Test basic configuration
        self.assertIsNotNone(task.instructions)
        self.assertTrue(len(task.instructions) > 0)
        self.assertEqual(task.response_format, TranslatedString)
        self.assertEqual(task.temperature, 0.0)
        self.assertEqual(task.top_p, 1.0)

    def test_multilingual_translation_task_instructions(self):
        """Test that translation task has appropriate instructions."""
        instructions = MULTILINGUAL_TRANSLATION.instructions
        
        # Should contain translation-related keywords
        self.assertIn("translate", instructions.lower())

    def test_translated_string_model_fields(self):
        """Test that TranslatedString has expected language fields."""
        # Create a sample instance to verify field structure
        sample_data = {
            # Germanic languages
            "en": "Hello",
            "de": "Hallo", 
            "nl": "Hallo",
            "sv": "Hej",
            "da": "Hej",
            "no": "Hei",
            
            # Romance languages
            "es": "Hola",
            "fr": "Bonjour",
            "it": "Ciao",
            "pt": "Olá",
            "ro": "Salut",
            "ca": "Hola",
            
            # Slavic languages
            "ru": "Привет",
            "pl": "Cześć",
            "cs": "Ahoj",
            "sk": "Ahoj", 
            "uk": "Привіт",
            "bg": "Здравей",
            "hr": "Bok",
            "sr": "Здраво",
            
            # East Asian languages
            "ja": "こんにちは",
            "ko": "안녕하세요",
            "zh": "你好",
            "zh_tw": "你好",
            
            # South Asian languages
            "hi": "नमस्ते",
            "bn": "হ্যালো",
            "te": "హలో",
            "ta": "வணக்கம்",
            "ur": "سلام",
            
            # Southeast Asian languages
            "th": "สวัสดี",
            "vi": "Xin chào",
            "id": "Halo",
            "ms": "Hello",
            "tl": "Kumusta",
            
            # Middle Eastern languages
            "ar": "مرحبا",
            "he": "שלום",
            "fa": "سلام",
            "tr": "Merhaba",
            
            # African languages
            "sw": "Hujambo",
            "am": "ሰላም",
            
            # Other European languages
            "fi": "Hei",
            "hu": "Szia",
            "et": "Tere",
            "lv": "Sveiki",
            "lt": "Labas",
            "el": "Γεια",
            
            # Nordic and other languages
            "is_": "Halló",
            "eu": "Kaixo",
            "cy": "Helo",
            "ga": "Dia dhuit",
            "mt": "Bonġu"
        }
        
        # This should not raise any validation errors
        translated_string = TranslatedString(**sample_data)
        
        # Verify some key fields
        self.assertEqual(translated_string.en, "Hello")
        self.assertEqual(translated_string.ja, "こんにちは")
        self.assertEqual(translated_string.es, "Hola")
        self.assertEqual(translated_string.zh, "你好")

    def test_translated_string_required_fields(self):
        """Test that TranslatedString requires all language fields."""
        # Try to create TranslatedString with missing fields - should fail
        with self.assertRaises(Exception):  # Pydantic ValidationError
            TranslatedString(en="Hello", ja="こんにちは")  # Missing many required fields

    def test_translated_string_field_types(self):
        """Test that TranslatedString fields are string types."""
        # Get model fields information
        fields = TranslatedString.model_fields
        
        # Check a few key fields are strings
        self.assertIn('en', fields)
        self.assertIn('ja', fields) 
        self.assertIn('es', fields)
        self.assertIn('zh', fields)
        
        # All fields should have string annotation
        for field_name, field_info in fields.items():
            # The annotation should be str
            self.assertEqual(field_info.annotation, str)


if __name__ == '__main__':
    unittest.main()
