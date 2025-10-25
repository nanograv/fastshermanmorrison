# Release Guide for FastShermanMorrison

This document outlines the step-by-step process to release a new version of the FastShermanMorrison package.

## Prerequisites

- Ensure you have `twine` installed: `pip install twine`
- Make sure you have PyPI credentials configured
- Verify all tests pass before starting

## Release Process

### 1. Update Version Information

**Important**: This project uses `setuptools_scm` for automatic versioning based on git tags.

- **Create a git tag** for the new version:
  ```bash
  git tag -a v0.5.4 -m "Release version 0.5.4"
  git push origin v0.5.4
  ```

- **Verify version update**: Check that `fastshermanmorrison/_version.py` gets updated automatically
- **Update README.md**: Update the version number in the citation section if needed

### 2. Update Documentation

- **Update CHANGELOG.md** (create if it doesn't exist):
  ```markdown
  ## [0.5.4] - 2024-01-XX
  
  ### Added
  - New features
  
  ### Changed
  - Changes to existing functionality
  
  ### Fixed
  - Bug fixes
  ```

- **Update README.md**: Ensure installation instructions and citation information are current
- **Update CITATION.cff**: Update version and date-released fields

### 3. Run Tests and Quality Checks

```bash
# Run tests
python -m pytest tests/

# Check code style (if you have flake8 configured)
flake8 fastshermanmorrison/

# Build locally to test
python setup.py build_ext --inplace
```

### 4. Clean Previous Builds

```bash
# Remove old build artifacts
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
```

### 5. Build Distribution Packages

```bash
# Build source distribution and wheel
python setup.py sdist bdist_wheel

# Verify the built packages
twine check dist/*
```

### 6. Upload to PyPI

```bash
# Upload to PyPI (test first with --repository testpypi if desired)
twine upload dist/*

# For test PyPI first:
# twine upload --repository testpypi dist/*
```

### 7. Create GitHub Release

- **Navigate to**: https://github.com/nanograv/fastshermanmorrison/releases
- **Click**: "Create a new release"
- **Choose a tag**: Select the tag you created (e.g., `v0.5.4`)
- **Release title**: `FastShermanMorrison v0.5.4`
- **Description**: Copy content from CHANGELOG.md for this version
- **Attach files**: Upload the `.tar.gz` and `.whl` files from `dist/`
- **Publish release**

### 8. Update Conda Package (if applicable)

If you maintain a conda package:

```bash
# Update conda recipe version
# Edit conda/meta.yaml or recipe/meta.yaml

# Build conda package locally to test
conda-build conda/

# Submit to conda-forge (if applicable)
```

### 9. Verify Installation

Test the new release:

```bash
# Test PyPI installation
pip install --upgrade fastshermanmorrison-pulsar

# Test GitHub installation
pip install git+https://github.com/nanograv/fastshermanmorrison.git@v0.5.4
```

### 10. Post-Release Tasks

- **Monitor**: Check for any issues or bug reports
- **Update**: Any downstream dependencies or documentation that references version numbers
- **Announce**: Notify users through appropriate channels (mailing lists, forums, etc.)

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR** (X.0.0): Breaking changes
- **MINOR** (X.Y.0): New features, backward compatible
- **PATCH** (X.Y.Z): Bug fixes, backward compatible

## Troubleshooting

### Common Issues

1. **Version not updating**: Ensure git tag is pushed and `setuptools_scm` can find it
2. **Build failures**: Check that Cython and numpy are properly installed
3. **Upload failures**: Verify PyPI credentials and package name uniqueness
4. **Import errors**: Test the built package locally before uploading

### Rollback Procedure

If a release has issues:

1. **Remove PyPI package**: Contact PyPI support to remove the problematic version
2. **Delete GitHub release**: Go to releases page and delete the release
3. **Remove git tag**: `git tag -d v0.5.4 && git push origin :refs/tags/v0.5.4`
4. **Fix issues** and repeat release process

## Notes

- This project uses `setuptools_scm` for automatic versioning - version numbers come from git tags
- The package name on PyPI is `fastshermanmorrison-pulsar`
- Cython extensions require proper build environment setup
- Always test locally before uploading to PyPI
