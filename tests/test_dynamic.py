from __future__ import annotations

from enum import Enum
from typing import get_args, get_origin

import pytest

from openaivec._dynamic import _MAX_ENUM_VALUES, EnumSpec, FieldSpec, ObjectSpec, _build_model

# ----------------------------- Success Cases -----------------------------


def test_build_model_primitives_and_arrays():
    spec = ObjectSpec(
        name="SampleObject",
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
        name="StatusContainer",
        fields=[
            FieldSpec(
                name="status_code",
                type="enum",
                description="status enum",
                enum_spec=EnumSpec(name="StatusCode", values=["ok", "error", "warn"]),
            ),
            FieldSpec(
                name="statuses",
                type="enum_array",
                description="array enum",
                enum_spec=EnumSpec(name="StatusCode", values=["ok", "error", "warn"]),
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


def test_build_model_enum_custom_name():
    spec = ObjectSpec(
        name="CustomEnumContainer",
        fields=[
            FieldSpec(
                name="status_code",
                type="enum",
                description="status enum",
                enum_spec=EnumSpec(name="StatusCodeEnum", values=["ok", "error"]),
            )
        ],
    )
    Model = _build_model(spec)
    mf = Model.model_fields
    enum_type = mf["status_code"].annotation
    assert enum_type.__name__ == "StatusCodeEnum"


def test_build_model_enum_case_insensitive_dedup():
    spec = ObjectSpec(
        name="EnumDedup",
        fields=[
            FieldSpec(
                name="mixed",
                type="enum",
                description="mixed case variants",
                enum_spec=EnumSpec(name="Mixed", values=["ok", "OK", "Ok"]),
            )
        ],
    )
    Model = _build_model(spec)
    enum_type = Model.model_fields["mixed"].annotation
    members = {m.name for m in enum_type}
    assert members == {"OK"}


def test_build_model_enum_size_boundary():
    values = [f"v{i}" for i in range(_MAX_ENUM_VALUES)]
    spec = ObjectSpec(
        name="EnumBoundary",
        fields=[
            FieldSpec(
                name="codes",
                type="enum",
                description="boundary size",
                enum_spec=EnumSpec(name="Codes", values=values),
            )
        ],
    )
    Model = _build_model(spec)
    enum_type = Model.model_fields["codes"].annotation
    assert len(list(enum_type)) == _MAX_ENUM_VALUES


@pytest.mark.parametrize("bad_name", ["lowercase", "1StartsWithDigit", "Has-Hyphen", "With Space", "Camel_Case"])
def test_build_model_enum_custom_name_invalid(bad_name: str):
    spec = ObjectSpec(
        name="CustomEnumContainer",
        fields=[
            FieldSpec(
                name="status_code",
                type="enum",
                description="status enum",
                enum_spec=EnumSpec(name=bad_name, values=["ok", "error"]),
            )
        ],
    )
    with pytest.raises(ValueError) as ei:
        _build_model(spec)
    assert "enum_spec.name" in str(ei.value)


def test_build_model_nested_object_and_object_array():
    address_spec = ObjectSpec(
        name="Address",
        fields=[
            FieldSpec(name="line1", type="string", description="line1"),
            FieldSpec(name="zip_code", type="string", description="zip"),
        ],
    )
    parent_spec = ObjectSpec(
        name="UserProfile",
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


# ----------------------------- Error Cases (MECE, unified) -----------------------------


@pytest.mark.parametrize(
    "fields,expected_substring,case_label",
    [
        # --- enum errors ---
        (
            [FieldSpec(name="a", type="enum", description="", enum_spec=None)],
            "enum type requires",
            "enum_missing_values",
        ),
        (
            [
                FieldSpec(
                    name="a",
                    type="enum",
                    description="",
                    enum_spec=EnumSpec(name="A", values=["x"]),
                    object_spec=ObjectSpec(name="O", fields=[]),
                )
            ],
            "must not provide object_spec",
            "enum_with_object_spec",
        ),
        (
            [
                FieldSpec(
                    name="a",
                    type="enum",
                    description="",
                    enum_spec=EnumSpec(name="A", values=[f"v{i}" for i in range(_MAX_ENUM_VALUES + 1)]),
                )
            ],
            "supports at most",
            "enum_size_overflow",
        ),
        (
            [FieldSpec(name="a", type="enum", description="", enum_spec=EnumSpec(name="badName", values=["x"]))],
            "enum_spec.name",
            "enum_invalid_name_pattern",
        ),
        # --- enum_array errors ---
        (
            [FieldSpec(name="a", type="enum_array", description="", enum_spec=None)],
            "enum_array type requires",
            "enum_array_missing_values",
        ),
        (
            [
                FieldSpec(
                    name="a",
                    type="enum_array",
                    description="",
                    enum_spec=EnumSpec(name="A", values=["x"]),
                    object_spec=ObjectSpec(name="O", fields=[]),
                )
            ],
            "enum_array type must not provide object_spec",
            "enum_array_with_object_spec",
        ),
        (
            [
                FieldSpec(
                    name="a",
                    type="enum_array",
                    description="",
                    enum_spec=EnumSpec(name="A", values=[f"v{i}" for i in range(_MAX_ENUM_VALUES + 1)]),
                )
            ],
            "supports at most",
            "enum_array_size_overflow",
        ),
        (
            [FieldSpec(name="a", type="enum_array", description="", enum_spec=EnumSpec(name="badName", values=["x"]))],
            "enum_spec.name",
            "enum_array_invalid_name_pattern",
        ),
        # --- object / object_array missing spec ---
        (
            [FieldSpec(name="a", type="object", description="")],
            "object type requires object_spec",
            "object_missing_spec",
        ),
        (
            [FieldSpec(name="a", type="object_array", description="")],
            "object_array type requires object_spec",
            "object_array_missing_spec",
        ),
        # --- object/object_array invalid nested name ---
        (
            [FieldSpec(name="a", type="object", description="", object_spec=ObjectSpec(name="badName", fields=[]))],
            "object_spec.name",
            "object_invalid_nested_name",
        ),
        (
            [
                FieldSpec(
                    name="a", type="object_array", description="", object_spec=ObjectSpec(name="badName", fields=[])
                )
            ],
            "object_spec.name",
            "object_array_invalid_nested_name",
        ),
        # --- object/object_array with enum_spec misuse ---
        (
            [
                FieldSpec(
                    name="a",
                    type="object",
                    description="",
                    enum_spec=EnumSpec(name="A", values=["x"]),
                    object_spec=ObjectSpec(name="O", fields=[]),
                )
            ],
            "must not define enum_spec",
            "object_with_enum_values",
        ),
        (
            [
                FieldSpec(
                    name="a",
                    type="object_array",
                    description="",
                    enum_spec=EnumSpec(name="A", values=["x"]),
                    object_spec=ObjectSpec(name="O", fields=[]),
                )
            ],
            "must not define enum_spec",
            "object_array_with_enum_values",
        ),
        # --- primitive misuses ---
        (
            [FieldSpec(name="a", type="string", description="", enum_spec=EnumSpec(name="A", values=["x"]))],
            "must not define enum_spec",
            "primitive_with_enum_values",
        ),
        (
            [FieldSpec(name="a", type="integer", description="", object_spec=ObjectSpec(name="O", fields=[]))],
            "must not define object_spec",
            "primitive_with_object_spec",
        ),
        # --- field name issues ---
        (
            [
                FieldSpec(name="dup", type="string", description=""),
                FieldSpec(name="dup", type="integer", description=""),
            ],
            "unique",
            "duplicate_names",
        ),
        ([FieldSpec(name="", type="string", description="")], "lower_snake_case", "empty_name"),
        ([FieldSpec(name="BadName", type="string", description="")], "lower_snake_case", "uppercase_name"),
    ],
)
def test_build_model_error_cases(fields: list[FieldSpec], expected_substring: str, case_label: str):  # noqa: D401
    """All invalid field configurations should raise ValueError with indicative message."""
    spec = ObjectSpec(name="Bad", fields=fields)
    with pytest.raises(ValueError) as ei:
        _build_model(spec)
    msg = str(ei.value)
    assert expected_substring in msg, f"case={case_label}, got={msg}"


# ----------------------------- Additional Coverage (previously uncovered) -----------------------------


@pytest.mark.parametrize(
    "field_type,field_name,object_name,num_values,should_raise,expected_error",
    [
        ("enum", "boundary", "EnumBoundary", _MAX_ENUM_VALUES, False, None),
        ("enum", "overflow", "EnumOverflow", _MAX_ENUM_VALUES + 1, True, "supports at most"),
        (
            "enum_array",
            "overflow_list",
            "EnumArrayOverflow",
            _MAX_ENUM_VALUES + 1,
            True,
            "enum_array type supports at most",
        ),
    ],
)
def test_enum_size_boundaries(field_type, field_name, object_name, num_values, should_raise, expected_error):
    """Test enum size boundaries for both enum and enum_array types."""
    values = [f"v{i}" for i in range(num_values)]
    spec = ObjectSpec(
        name=object_name,
        fields=[
            FieldSpec(
                name=field_name,
                type=field_type,
                description="size test",
                enum_spec=EnumSpec(name="TestEnum", values=values),
            )
        ],
    )

    if should_raise:
        with pytest.raises(ValueError) as ei:
            _build_model(spec)
        assert expected_error in str(ei.value)
    else:
        Model = _build_model(spec)
        enum_type = Model.model_fields[field_name].annotation
        if field_type == "enum":
            assert len(list(enum_type)) == _MAX_ENUM_VALUES
        else:  # enum_array
            from typing import get_args, get_origin

            assert get_origin(enum_type) is list
            inner_enum = get_args(enum_type)[0]
            assert len(list(inner_enum)) == _MAX_ENUM_VALUES


def test_enum_array_invalid_name_pattern():
    spec = ObjectSpec(
        name="EnumArrayInvalidName",
        fields=[
            FieldSpec(
                name="vals",
                type="enum_array",
                description="bad name",
                enum_spec=EnumSpec(name="badName", values=["a", "b"]),
            )
        ],
    )
    with pytest.raises(ValueError) as ei:
        _build_model(spec)
    assert "enum_spec.name" in str(ei.value)


@pytest.mark.parametrize(
    "field_type,field_name,object_name,bad_nested_name",
    [
        ("object", "child", "RootBadObject", "bad_name"),
        ("object_array", "children", "RootBadObjectArray", "1Bad"),
    ],
)
def test_object_spec_invalid_names(field_type, field_name, object_name, bad_nested_name):
    """Test that invalid nested object spec names are rejected."""
    spec = ObjectSpec(
        name=object_name,
        fields=[
            FieldSpec(
                name=field_name,
                type=field_type,
                description="invalid nested name test",
                object_spec=ObjectSpec(name=bad_nested_name, fields=[]),
            )
        ],
    )
    with pytest.raises(ValueError) as ei:
        _build_model(spec)
    assert "object_spec.name" in str(ei.value)


def test_object_array_with_enum_spec():
    spec = ObjectSpec(
        name="ObjectArrayWithEnumSpec",
        fields=[
            FieldSpec(
                name="children",
                type="object_array",
                description="invalid enum_spec",
                enum_spec=EnumSpec(name="E", values=["x"]),
                object_spec=ObjectSpec(name="Child", fields=[]),
            )
        ],
    )
    with pytest.raises(ValueError) as ei:
        _build_model(spec)
    assert "must not define enum_spec" in str(ei.value)


def test_enum_case_insensitive_dedup():
    spec = ObjectSpec(
        name="EnumDedup",
        fields=[
            FieldSpec(
                name="code",
                type="enum",
                description="dedup",
                enum_spec=EnumSpec(name="CodeEnum", values=["ok", "OK", "Ok"]),
            )
        ],
    )
    Model = _build_model(spec)
    enum_type = Model.model_fields["code"].annotation
    # After upper + set, only one member expected
    assert len(list(enum_type)) == 1
    assert list(enum_type)[0].name == "OK"
