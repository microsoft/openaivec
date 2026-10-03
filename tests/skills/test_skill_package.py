import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "skills/openaivec-skill"
WORKFLOW = ROOT / ".github/workflows/publish-skill.yml"


def test_release_manifest_and_curated_notes_match_skill():
    text = (SKILL / "SKILL.md").read_text()
    frontmatter = text.split("---", 2)[1]
    version = re.search(r'^  version: "(\d+\.\d+\.\d+)"$', frontmatter, re.MULTILINE)
    assert version is not None
    description = re.search(r"^description: (.+)$", frontmatter, re.MULTILINE)
    assert description is not None and len(description.group(1)) <= 1024
    assert "Proactively" in description.group(1)
    assert "repository: https://github.com/microsoft/openaivec" in frontmatter
    notes = (SKILL / "RELEASE_NOTES.md").read_text()
    assert notes.startswith(f"# openaivec-skill {version.group(1)} ")
    assert "Business-user installation" in notes
    assert "pin installation to that exact tag" in notes
    workflow = WORKFLOW.read_text()
    assert '--notes-file "$SKILL_RELEASE_NOTES"' in workflow
    assert "--generate-notes" not in workflow
    manifest = re.search(r"      SKILL_PACKAGE_FILES: \|\n((?:        \S.*\n)+)", workflow)
    assert manifest is not None
    files = [line.strip() for line in manifest.group(1).splitlines()]
    assert len(files) == len(set(files))
    assert {"GETTING_STARTED.md", "RELEASE_NOTES.md", "scripts/bulk_runner.py"} <= set(files)
    assert all((SKILL / file).is_file() for file in files)


@pytest.mark.parametrize("path", sorted(SKILL.rglob("*.md")))
def test_portable_documentation_links_resolve(path):
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", path.read_text()):
        if "://" in target or target.startswith("#"):
            continue
        assert (path.parent / target.split("#", 1)[0]).is_file(), f"{path.name}: {target}"


@pytest.mark.parametrize("name", ["GETTING_STARTED.md", "RELEASE_NOTES.md"])
def test_business_guides_do_not_require_code_or_terminal_steps(name):
    text = (SKILL / name).read_text()
    assert not re.search(r"```(?:bash|sh|python|sql)\b", text)
    assert not re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", text)
    assert "openaivec-skill-vX.Y.Z" in text
    assert "```text" in text


def test_assistant_snippets_compile_and_helper_has_no_private_api_imports():
    for name in ("performance-and-execution.md", "duckdb-workflows.md"):
        text = (SKILL / "references" / name).read_text()
        for index, code in enumerate(re.findall(r"```python\n(.*?)```", text, re.DOTALL)):
            compile(code, f"{name}:example-{index}", "exec")
    module = ast.parse((SKILL / "scripts/bulk_runner.py").read_text())
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("openaivec."):
            assert not any(part.startswith("_") for part in node.module.split(".")[1:])


def test_evaluation_cases_are_unique_and_well_formed():
    triggers = json.loads((SKILL / "evals/trigger-cases.json").read_text())
    assert len({case["query"] for case in triggers}) == len(triggers)
    assert all(isinstance(case["should_trigger"], bool) for case in triggers)
    assert any(case["should_trigger"] and "このExcel" in case["query"] for case in triggers)
    assert any(not case["should_trigger"] and "数値だけ" in case["query"] for case in triggers)
    workflows = json.loads((SKILL / "evals/workflow-cases.json").read_text())
    assert len({case["id"] for case in workflows}) == len(workflows)
    assert {"stateless-harness", "accepted-pilot-reuse", "business-user-install"} <= {case["id"] for case in workflows}
    assert all(case["request"] and case["required"] and case["forbidden"] for case in workflows)
