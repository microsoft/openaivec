from .dependency_parsing import DEPENDENCY_PARSING
from .keyword_extraction import KEYWORD_EXTRACTION
from .morphological_analysis import MORPHOLOGICAL_ANALYSIS
from .named_entity_recognition import NAMED_ENTITY_RECOGNITION
from .sentiment_analysis import SENTIMENT_ANALYSIS
from .translation import MULTILINGUAL_TRANSLATION

__all__ = [
    "MULTILINGUAL_TRANSLATION",
    "MORPHOLOGICAL_ANALYSIS",
    "NAMED_ENTITY_RECOGNITION",
    "SENTIMENT_ANALYSIS",
    "DEPENDENCY_PARSING",
    "KEYWORD_EXTRACTION",
]
