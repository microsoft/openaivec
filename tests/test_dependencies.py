from importlib.metadata import metadata, requires

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


@pytest.mark.parametrize(
    ("python_version", "supported"),
    [("3.10.0", False), ("3.11.0", True), ("3.12.0", True)],
)
def test_minimum_python_version(python_version: str, supported: bool) -> None:
    requirement = metadata("openaivec")["Requires-Python"]
    assert requirement == ">=3.11"
    assert SpecifierSet(requirement).contains(python_version) is supported


def test_runtime_imports_have_direct_dependencies():
    dependencies = {
        requirement.name.lower().replace("_", "-"): requirement
        for entry in requires("openaivec") or []
        if (requirement := Requirement(entry)).marker is None
    }

    assert {"httpx", "numpy", "pydantic", "typing-extensions"} <= dependencies.keys()
    assert not dependencies["pydantic"].specifier.contains("1.10.0")
    assert "pyspark" not in dependencies
    assert "synapseml" not in dependencies


def test_aiohttp_dependency_excludes_missing_timeout_exceptions():
    dependency = next(
        requirement for entry in requires("openaivec") or [] if (requirement := Requirement(entry)).name == "aiohttp"
    )

    assert not dependency.specifier.contains("3.9.3")


@pytest.mark.parametrize(
    ("package_name", "vulnerable_version", "patched_version"),
    [
        ("certifi", "2026.7.22", "2026.10.18"),
        ("pyjwt", "2.13.0", "2.15.0"),
        ("pyjwt", "2.14.0", "2.15.0"),
        ("urllib3", "1.26.0", "2.8.0"),
        ("urllib3", "2.7.0", "2.8.0"),
    ],
)
def test_security_dependencies_exclude_vulnerable_versions(
    package_name: str, vulnerable_version: str, patched_version: str
) -> None:
    dependency = next(
        requirement
        for entry in requires("openaivec") or []
        if (requirement := Requirement(entry)).name.lower() == package_name
    )

    assert not dependency.specifier.contains(vulnerable_version)
    assert dependency.specifier.contains(patched_version)
