from __future__ import annotations

from enum import Enum
from typing import get_args, get_origin

import pytest

from openaivec._dynamic import FieldSpec, ObjectSpec, _build_model

# ----------------------------- Success Cases -----------------------------


def test_build_model_primitives_and_arrays():
    spec = ObjectSpec(
        name="sample_object",
        fields=[
            FieldSpec(name="title", type="string", description="a title"),
            FieldSpec(name="count", type="integer", description="an int"),
            FieldSpec(name="score", type="float", description="a float"),
            FieldSpec(name="flag", type="boolean", description="a bool"),
            FieldSpec(name="tags", type="string_array", description="list of strings"),
            FieldSpec(name="nums", type="integer_array", description="list of ints"),
            FieldSpec(name="scores", type="float_array", description="list of floats"),
            FieldSpec(name="flags", type="boolean_array", description="list of bools"),
        ],
    )
    Model = _build_model(spec)

    # Class name UpperCamel
    assert Model.__name__ == "SampleObject"

    mf = Model.model_fields
    assert mf["title"].annotation is str
    assert mf["count"].annotation is int
    assert mf["score"].annotation is float
    assert mf["flag"].annotation is bool

    # Array annotations
    for name, inner in [
        ("tags", str),
        ("nums", int),
        ("scores", float),
        ("flags", bool),
    ]:
        ann = mf[name].annotation
        assert get_origin(ann) in (list, list[str].__origin__ if hasattr(list[str], "__origin__") else list)  # type: ignore[attr-defined]
        assert get_args(ann)[0] is inner

    # Description metadata retained
    assert mf["title"].description == "a title"
    assert mf["flags"].description == "list of bools"


def test_build_model_enum_and_enum_array():
    spec = ObjectSpec(
        name="status_container",
        fields=[
            FieldSpec(name="status_code", type="enum", description="status enum", enum_values=["ok", "error", "warn"]),
            FieldSpec(
                name="statuses", type="enum_array", description="array enum", enum_values=["ok", "error", "warn"]
            ),
        ],
    )
    Model = _build_model(spec)
    mf = Model.model_fields

    enum_type = mf["status_code"].annotation
    assert isinstance(enum_type, type) and issubclass(enum_type, Enum)
    assert enum_type.__name__ == "StatusCode"
    # Members uppercased
    member_names = {m.name for m in enum_type}
    assert {"OK", "ERROR", "WARN"}.issubset(member_names)

    enum_list_ann = mf["statuses"].annotation
    assert get_origin(enum_list_ann) is list
    (inner_enum,) = get_args(enum_list_ann)
    assert issubclass(inner_enum, Enum)


def test_build_model_nested_object_and_object_array():
    address_spec = ObjectSpec(
        name="address",
        fields=[
            FieldSpec(name="line1", type="string", description="line1"),
            FieldSpec(name="zip_code", type="string", description="zip"),
        ],
    )
    parent_spec = ObjectSpec(
        name="user_profile",
        fields=[
            FieldSpec(name="name", type="string", description="username"),
            FieldSpec(name="address", type="object", description="address obj", object_spec=address_spec),
            FieldSpec(
                name="previous_addresses", type="object_array", description="prev addresses", object_spec=address_spec
            ),
        ],
    )
    Model = _build_model(parent_spec)
    mf = Model.model_fields

    nested_model_type = mf["address"].annotation
    assert nested_model_type.__name__ == "Address"
    assert {"line1", "zip_code"}.issubset(nested_model_type.model_fields.keys())

    addresses_ann = mf["previous_addresses"].annotation
    assert get_origin(addresses_ann) is list
    (inner_type,) = get_args(addresses_ann)
    assert inner_type.__name__ == "Address"


# ----------------------------- Error Cases (MECE) -----------------------------


@pytest.mark.parametrize(
    "field_spec,expected_substring",
    [
        # 1. enum missing enum_values
        (FieldSpec(name="a", type="enum", description="", enum_values=None), "enum type requires"),
        # 2. enum with object_spec
        (
            FieldSpec(
                name="a", type="enum", description="", enum_values=["x"], object_spec=ObjectSpec(name="o", fields=[])
            ),
            "must not provide object_spec",
        ),
        # 3. enum_array missing enum_values
        (FieldSpec(name="a", type="enum_array", description="", enum_values=None), "enum_array type requires"),
        # 4. enum_array with object_spec
        (
            FieldSpec(
                name="a",
                type="enum_array",
                description="",
                enum_values=["x"],
                object_spec=ObjectSpec(name="o", fields=[]),
            ),
            "enum_array type must not provide object_spec",
        ),
        # 5. object missing object_spec
        (FieldSpec(name="a", type="object", description=""), "object type requires object_spec"),
        # 6. object_array missing object_spec
        (FieldSpec(name="a", type="object_array", description=""), "object_array type requires object_spec"),
        # 7. object with enum_values (invalid)
        (
            FieldSpec(
                name="a", type="object", description="", enum_values=["x"], object_spec=ObjectSpec(name="o", fields=[])
            ),
            "must not define enum_values",
        ),
        # 8. primitive with enum_values
        (FieldSpec(name="a", type="string", description="", enum_values=["x"]), "must not define enum_values"),
        # 9. primitive with object_spec
        (
            FieldSpec(name="a", type="integer", description="", object_spec=ObjectSpec(name="o", fields=[])),
            "must not define object_spec",
        ),
    ],
)
def test_build_model_error_cases(field_spec: FieldSpec, expected_substring: str):
    spec = ObjectSpec(name="bad", fields=[field_spec])
    with pytest.raises(ValueError) as ei:
        _build_model(spec)
    assert expected_substring in str(ei.value)
