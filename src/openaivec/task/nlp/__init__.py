"""Natural language processing task factories."""

from .dependency_parsing import dependency_parsing
from .keyword_extraction import keyword_extraction
from .morphological_analysis import morphological_analysis
from .named_entity_recognition import named_entity_recognition
from .sentiment_analysis import sentiment_analysis
from .translation import multilingual_translation

__all__ = [
    "dependency_parsing",
    "keyword_extraction",
    "morphological_analysis",
    "multilingual_translation",
    "named_entity_recognition",
    "sentiment_analysis",
]

