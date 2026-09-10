from importlib.metadata import requires

from packaging.requirements import Requirement


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
