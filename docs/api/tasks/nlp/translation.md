# Translation Task

`multilingual_translation(target_languages=["ja"])` requests only Japanese
and produces a schema with only the required `ja` field. Pass multiple supported
language codes to request several translations; omitting `target_languages`
preserves the original all-51-language response model (`TranslatedString`).
Empty, duplicate, and unknown codes are rejected.

The Icelandic JSON key is `is`: access it in Python as `result.is_`, and use
`result.model_dump(by_alias=True)` to serialize language-code keys. The
full model still accepts `is_` when validating older Python-side data.

::: openaivec.task.nlp.translation
