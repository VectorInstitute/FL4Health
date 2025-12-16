# FL4Health Documentation

This directory contains the source files for FL4Health documentation, built with mkdocs and Material for MkDocs.

## Building Documentation Locally

1. Install documentation dependencies:
   ```bash
   uv sync --group docs
   ```

2. Serve documentation with live reload:
   ```bash
   uv run mkdocs serve
   ```

3. Open http://127.0.0.1:8000 in your browser

## Building for Production

```bash
uv run mkdocs build --strict
```

The generated site will be in the `site/` directory.

## Documentation Structure

- `docs/index.md` - Home page
- `docs/quickstart.md` - Quick start guide
- `docs/module_guides/` - Module documentation
- `docs/examples/` - Example documentation
- `docs/api.md` - API reference (auto-generated from docstrings using mkdocstrings)
- `docs/contributing.md` - Contributing guide
- `docs/assets/` - Images, logos, and static assets
- `docs/stylesheets/` - Custom CSS (Vector Institute branding)
- `docs/overrides/` - Theme customization templates

## Configuration

Documentation configuration is in `mkdocs.yml` at the repository root.
