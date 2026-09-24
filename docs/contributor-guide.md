# Contributor Guidelines

Refer to [AGENTS.md](https://github.com/microsoft/openaivec/blob/main/AGENTS.md) in the repository root for the authoritative contributor guide.

## Validate documentation

Install the documentation dependencies and check for broken links, malformed
docstrings, and other MkDocs warnings before submitting documentation changes:

```bash
uv sync --group docs
uv run mkdocs build --strict -d site
```

The generated `site/` directory is gitignored. The Pages workflow runs the
same strict build for documentation pull requests without uploading or
deploying them. Tagged builds publish only after the strict build succeeds.

## Publishing a package release

Once the intended changes are merged into `main`, create and push a `vX.Y.Z`
tag pointing to that commit. The PyPI workflow checks the tagged commit on
Python 3.10-3.12 with Ruff, Pyright, the full test suite, and a distribution
build before publishing. The `integration` environment must provide
`OPENAI_API_KEY` for the full suite; the `pypi` environment supplies the
trusted-publishing configuration. After validation, the workflow signs and
attaches the distributions to a GitHub Release with generated release notes,
then publishes them to PyPI. Confirm that both the workflow and the published
package version succeeded before announcing the release.
