# Changelog

## 0.1.15 — 2026-10-31

### Added

- Support for running the CLI with `python -m fastdrs`.
- Regression coverage for checkpoint preprocessing serialization and self-defining LiteRT export.

### Fixed

- Preserve non-default image sizes when deserializing checkpoint preprocessing.
- Use the saved `img_size` field when building inference transforms.
- Detect self-defining checkpoints from the input path during LiteRT export.

### Documentation

- Document model listing, checkpoint inspection, prediction, JSON output, and LiteRT export in the PyPI README.
- Correct LiteRT extra installation instructions and checkpoint loading examples.
