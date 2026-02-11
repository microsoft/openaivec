"""Shared prompt-building utilities for curated task definitions."""

from collections.abc import Mapping, Sequence

__all__ = []


def join_sections(*sections: str) -> str:
    """Join non-empty prompt sections with a blank line."""
    cleaned = [section.strip() for section in sections if section and section.strip()]
    return "\n\n".join(cleaned)


def same_language_policy() -> str:
    """Return the shared policy for multilingual natural-language fields."""
    return (
        "Provide natural-language fields in the same language as the input text. "
        "Do not switch explanation language unless the task explicitly requires translation."
    )


def english_categorical_policy(categorical_fields: Sequence[str]) -> str:
    """Return policy text for categorical fields that must stay in English."""
    fields = ", ".join(categorical_fields)
    return (
        "IMPORTANT: Keep predefined categorical values in exact English tokens. "
        f"The following fields must use English values only: {fields}."
    )


def mapping_section(title: str, mapping: Mapping[str, str]) -> str:
    """Render a bullet-style section from key-value mappings."""
    lines = [f"{title}:"]
    for key, value in mapping.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def grouped_mapping_section(title: str, mapping: Mapping[str, Sequence[str]]) -> str:
    """Render a bullet-style section from key to list-of-values mappings."""
    lines = [f"{title}:"]
    for key, values in mapping.items():
        joined_values = ", ".join(values)
        lines.append(f"- {key}: {joined_values}")
    return "\n".join(lines)

