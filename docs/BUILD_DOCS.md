# Regenerating Sphinx Documentation

This guide explains how to rebuild the KodeAgent documentation after making changes.

## Prerequisites

Install documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

Or install Sphinx manually:

```bash
pip install sphinx sphinx-rtd-theme myst-parser sphinx-autodoc-typehints
```

## Quick Build

### On Windows (PowerShell/CMD):

```bash
cd docs
sphinx-build -b html . _build/html
```

### On Linux/Mac:

```bash
cd docs
sphinx-build -b html . _build/html
```

## Using Make (if available)

If you have `make` installed:

```bash
cd docs
make html
```

## View the Documentation

After building, open the generated HTML:

**Windows**:
```bash
start _build/html/index.html
```

**Linux**:
```bash
xdg-open _build/html/index.html
```

**Mac**:
```bash
open _build/html/index.html
```

## Clean Build

To remove old build files and start fresh:

```bash
cd docs
rm -rf _build
sphinx-build -b html . _build/html
```

## Auto-rebuild on Changes

For development, use `sphinx-autobuild`:

```bash
pip install sphinx-autobuild
cd docs
sphinx-autobuild . _build/html
```

Then open http://localhost:8000 in your browser. The docs will auto-reload when you save changes!

## Common Issues

### Module Not Found

If you get "module not found" errors:

```bash
# Make sure kodeagent is installed
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%CD%\src          # Windows
```

### Missing Dependencies

```bash
pip install --upgrade sphinx sphinx-rtd-theme myst-parser
```

### Build Warnings

Warnings about missing references are usually safe to ignore during development.

## What Was Added

The new security documentation includes:

1. **security.rst** - Comprehensive security guide
   - Multi-layer security architecture
   - Block diagram of security flow
   - Pattern detection details
   - Risk scoring system
   - Best practices
   - Configuration guide
   - Testing instructions

2. **Updated index.rst** - Added Security section to main navigation

## Verifying the Build

After building, check that:

1. Security section appears in the sidebar
2. Block diagram renders correctly
3. Code examples have syntax highlighting
4. API references link properly
5. All internal links work

## Publishing

To publish to Read the Docs or GitHub Pages, commit the changes:

```bash
git add docs/security.rst docs/index.rst
git commit -m "Add comprehensive security documentation"
git push
```

Read the Docs will automatically rebuild on push.
