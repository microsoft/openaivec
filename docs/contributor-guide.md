# Contributor Guidelines

Refer to [AGENTS.md](https://github.com/microsoft/openaivec/blob/main/AGENTS.md) in the repository root for the authoritative contributor guide.

## Validate documentation

Install the documentation dependencies and check for broken links, malformed
docstrings, and other MkDocs warnings before submitting documentation changes:

```bash
uv sync --group docs
uv run mkdocs build --strict -d site
```

The generated `site/` directory is gitignored. The Pages workflow uses the
same strict configuration, so warning-producing builds cannot be published.
