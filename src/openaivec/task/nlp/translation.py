"""Multilingual translation task definition."""

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from openaivec._model import PreparedTask
from openaivec.task._prompt_templates import join_sections
from openaivec.task._registry import TaskSpec

__all__ = ["multilingual_translation"]


class TranslatedString(BaseModel):
    """Translations for a fixed set of language-code fields."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

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
    is_: str = Field(alias="is", description="Translated text in Icelandic")

    # Other languages
    eu: str = Field(description="Translated text in Basque")
    cy: str = Field(description="Translated text in Welsh")
    ga: str = Field(description="Translated text in Irish")
    mt: str = Field(description="Translated text in Maltese")


def _build_instructions(codes: Sequence[str]) -> str:
    return join_sections(
        f"Translate the input text into these target language codes: {', '.join(codes)}.",
        "Keep meaning, tone, and named entities consistent across languages.",
        "Return only the requested language-code JSON fields. Do not add explanations.",
    )


def multilingual_translation(target_languages: Sequence[str] | None = None) -> PreparedTask[BaseModel]:
    """Create a translation task for selected language codes.

    Args:
        target_languages (Sequence[str] | None): Language codes to translate into.
            ``None`` (the default) requests all 51 supported languages. Use ``is``
            for Icelandic; the Python attribute remains ``is_`` for compatibility.

    Returns:
        PreparedTask[BaseModel]: Task with exactly the requested response fields.

    Raises:
        TypeError: If target_languages is a string rather than a sequence.
        ValueError: If the selection is empty, duplicated, or unsupported.
    """
    supported = {field.alias or name: name for name, field in TranslatedString.model_fields.items()}
    if isinstance(target_languages, str):
        raise TypeError("target_languages must be a sequence of language codes, not a string")
    codes = tuple(supported) if target_languages is None else tuple(target_languages)
    if (
        not codes
        or any(not isinstance(code, str) or code not in supported for code in codes)
        or len(codes) != len(set(codes))
    ):
        raise ValueError("target_languages must contain unique supported language codes")
    response_format = TranslatedString
    if target_languages is not None:
        fields: dict[str, Any] = {
            supported[code]: (str, TranslatedString.model_fields[supported[code]]) for code in codes
        }
        response_format = create_model(
            "SelectedTranslations",
            __config__=ConfigDict(extra="forbid", populate_by_name=True),
            __module__=__name__,
            **fields,
        )
    return PreparedTask(
        instructions=_build_instructions(codes),
        response_format=response_format,
    )


TASK_SPEC = TaskSpec(
    key="nlp.multilingual_translation",
    domain="nlp",
    summary="Translate input text into selected languages (all by default).",
    factory=multilingual_translation,
    response_format=TranslatedString,
)
