"""Multilingual translation task definition."""

from pydantic import BaseModel, ConfigDict, Field

from openaivec._model import PreparedTask
from openaivec.task._prompt_templates import join_sections
from openaivec.task._registry import TaskSpec

__all__ = ["multilingual_translation"]


class TranslatedString(BaseModel):
    """Translations for a fixed set of language-code fields."""

    model_config = ConfigDict(extra="forbid")

    # Germanic languages
    en: str = Field(description="Translated text in English")
    de: str = Field(description="Translated text in German")
    nl: str = Field(description="Translated text in Dutch")
    sv: str = Field(description="Translated text in Swedish")
    da: str = Field(description="Translated text in Danish")
    no: str = Field(description="Translated text in Norwegian")

    # Romance languages
    es: str = Field(description="Translated text in Spanish")
    fr: str = Field(description="Translated text in French")
    it: str = Field(description="Translated text in Italian")
    pt: str = Field(description="Translated text in Portuguese")
    ro: str = Field(description="Translated text in Romanian")
    ca: str = Field(description="Translated text in Catalan")

    # Slavic languages
    ru: str = Field(description="Translated text in Russian")
    pl: str = Field(description="Translated text in Polish")
    cs: str = Field(description="Translated text in Czech")
    sk: str = Field(description="Translated text in Slovak")
    uk: str = Field(description="Translated text in Ukrainian")
    bg: str = Field(description="Translated text in Bulgarian")
    hr: str = Field(description="Translated text in Croatian")
    sr: str = Field(description="Translated text in Serbian")

    # East Asian languages
    ja: str = Field(description="Translated text in Japanese")
    ko: str = Field(description="Translated text in Korean")
    zh: str = Field(description="Translated text in Chinese (Simplified)")
    zh_tw: str = Field(description="Translated text in Chinese (Traditional)")

    # South Asian languages
    hi: str = Field(description="Translated text in Hindi")
    bn: str = Field(description="Translated text in Bengali")
    te: str = Field(description="Translated text in Telugu")
    ta: str = Field(description="Translated text in Tamil")
    ur: str = Field(description="Translated text in Urdu")

    # Southeast Asian languages
    th: str = Field(description="Translated text in Thai")
    vi: str = Field(description="Translated text in Vietnamese")
    id: str = Field(description="Translated text in Indonesian")
    ms: str = Field(description="Translated text in Malay")
    tl: str = Field(description="Translated text in Filipino")

    # Middle Eastern languages
    ar: str = Field(description="Translated text in Arabic")
    he: str = Field(description="Translated text in Hebrew")
    fa: str = Field(description="Translated text in Persian")
    tr: str = Field(description="Translated text in Turkish")

    # African languages
    sw: str = Field(description="Translated text in Swahili")
    am: str = Field(description="Translated text in Amharic")

    # Other European languages
    fi: str = Field(description="Translated text in Finnish")
    hu: str = Field(description="Translated text in Hungarian")
    et: str = Field(description="Translated text in Estonian")
    lv: str = Field(description="Translated text in Latvian")
    lt: str = Field(description="Translated text in Lithuanian")
    el: str = Field(description="Translated text in Greek")

    # Nordic language (name avoids Python keyword conflict)
    is_: str = Field(description="Translated text in Icelandic")

    # Other languages
    eu: str = Field(description="Translated text in Basque")
    cy: str = Field(description="Translated text in Welsh")
    ga: str = Field(description="Translated text in Irish")
    mt: str = Field(description="Translated text in Maltese")


def _build_instructions() -> str:
    return join_sections(
        "Translate the input text into all target languages defined by the response schema.",
        "Keep meaning, tone, and named entities consistent across languages.",
        "Return only the structured JSON fields. Do not add explanations.",
    )


def multilingual_translation() -> PreparedTask[TranslatedString]:
    """Create a multilingual translation task."""
    return PreparedTask(
        instructions=_build_instructions(),
        response_format=TranslatedString,
    )


TASK_SPEC = TaskSpec(
    key="nlp.multilingual_translation",
    domain="nlp",
    summary="Translate input text into a fixed set of languages.",
    factory=multilingual_translation,
    response_format=TranslatedString,
)

