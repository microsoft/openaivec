"""Schema inference package.

Internal helpers now live in :mod:`openaivec._schema.infer`; this module simply
re-exports the main entry points so ``from openaivec._schema import ...`` still
behaves the same."""

from .infer import SchemaInferenceInput, SchemaInferenceOutput, SchemaInferer

__all__ = ["SchemaInferenceOutput", "SchemaInferenceInput", "SchemaInferer"]
