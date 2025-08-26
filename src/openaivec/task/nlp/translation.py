"""Multilingual translation task for OpenAI API.

This module provides a predefined task that translates text into multiple languages
using OpenAI's language models. The translation covers a comprehensive set of
languages including Germanic, Romance, Slavic, East Asian, South Asian, Southeast
Asian, Middle Eastern, African, and other language families.

The task is designed to be used with the OpenAI API for batch processing and
provides structured output with consistent language code naming.

Example:
    Basic usage with BatchResponses:

    ```python
    from openai import OpenAI
    from openaivec._responses import BatchResponses
    from openaivec.task import nlp

    client = OpenAI()
    translator = BatchResponses.of_task(
        client=client,
        model_name="gpt-4.1-mini",
        task=nlp.MULTILINGUAL_TRANSLATION
    )

    texts = ["Hello", "Good morning", "Thank you"]
    translations = translator.parse(texts)

    for translation in translations:
        print(f"English: {translation.en}")
        print(f"Japanese: {translation.ja}")
        print(f"Spanish: {translation.es}")
    ```

    With pandas integration:

    ```python
    import pandas as pd
    from openaivec import pandas_ext  # Required for .ai accessor
    from openaivec.task import nlp

    df = pd.DataFrame({"text": ["Hello", "Goodbye"]})
    df["translations"] = df["text"].ai.task(nlp.MULTILINGUAL_TRANSLATION)

    # Extract specific languages
    extracted_df = df.ai.extract("translations")
    print(extracted_df[["text", "translations_en", "translations_ja", "translations_fr"]])
    ```

Attributes:
    MULTILINGUAL_TRANSLATION (PreparedTask): A prepared task instance configured
        for multilingual translation with temperature=0.0 and top_p=1.0 for
        deterministic output.

Note:
    The translation covers 58 languages across major language families. All field
    names use ISO 639-1 language codes where possible, with some exceptions like
    'zh_tw' for Traditional Chinese and 'is_' for Icelandic (to avoid Python
    keyword conflicts).

    Languages included:
    - Germanic: English, German, Dutch, Swedish, Danish, Norwegian, Icelandic
    - Romance: Spanish, French, Italian, Portuguese, Romanian, Catalan
    - Slavic: Russian, Polish, Czech, Slovak, Ukrainian, Bulgarian, Croatian, Serbian
    - East Asian: Japanese, Korean, Chinese (Simplified/Traditional)
    - South Asian: Hindi, Bengali, Telugu, Tamil, Urdu
    - Southeast Asian: Thai, Vietnamese, Indonesian, Malay, Filipino
    - Middle Eastern: Arabic, Hebrew, Persian, Turkish
    - African: Swahili, Amharic
    - Other European: Finnish, Hungarian, Estonian, Latvian, Lithuanian, Greek
    - Celtic: Welsh, Irish
    - Other: Basque, Maltese
"""

from pydantic import BaseModel, Field

from openaivec._model import PreparedTask

__all__ = ["MULTILINGUAL_TRANSLATION"]


class TranslatedString(BaseModel):
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

    # Nordic languages
    is_: str = Field(description="Translated text in Icelandic")

    # Other languages
    eu: str = Field(description="Translated text in Basque")
    cy: str = Field(description="Translated text in Welsh")
    ga: str = Field(description="Translated text in Irish")
    mt: str = Field(description="Translated text in Maltese")


instructions = "Translate the following text into multiple languages. "

MULTILINGUAL_TRANSLATION = PreparedTask(
    instructions=instructions, response_format=TranslatedString, api_kwargs={"temperature": 0.0, "top_p": 1.0}
)
