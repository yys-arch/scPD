# Pre-Release Checklist for scPD

## ✅ Completed

- [x] Remove all Chinese comments and documentation
- [x] Translate all docstrings to English
- [x] Clean up unnecessary verbose comments
- [x] Enhance README with badges and sections
- [x] Create CONTRIBUTING.md
- [x] Create .gitignore
- [x] Set up GitHub Actions CI
- [x] Create LICENSE file (MIT)
- [x] Update pyproject.toml metadata
- [x] Add Citation section to README
- [x] Add Support section to README

## 📝 TODO Before First Commit

### Critical (Must Do)
- [ ] Replace `[Your Name or Organization]` in LICENSE file
- [ ] Replace `yourusername` in all URLs:
  - [ ] README.md (3 locations)
  - [ ] pyproject.toml (4 locations)
  - [ ] CONTRIBUTING.md (1 location)
- [ ] Replace `your.email@example.com` in pyproject.toml
- [ ] Update Citation author name in README.md

### Recommended
- [ ] Add your actual GitHub username to all URLs
- [ ] Test installation: `pip install -e .`
- [ ] Run tests: `pytest tests/`
- [ ] Review all example scripts work
- [ ] Add a screenshot or plot to README

## 🚀 TODO Before PyPI Release

### Package Testing
- [ ] Test installation in clean virtual environment
- [ ] Test on Python 3.8, 3.9, 3.10, 3.11, 3.12
- [ ] Test on Linux, macOS, Windows (if possible)
- [ ] Verify all dependencies install correctly
- [ ] Run full test suite with coverage

### Documentation
- [ ] Verify all docstrings are complete
- [ ] Check README renders correctly on GitHub
- [ ] Test all code examples in README
- [ ] Ensure examples/ directory scripts run

### Version Control
- [ ] Create initial git repository
- [ ] Make first commit
- [ ] Push to GitHub
- [ ] Create v0.1.0 tag
- [ ] Write release notes

### PyPI Preparation
- [ ] Create PyPI account (if needed)
- [ ] Build package: `python -m build`
- [ ] Test upload to TestPyPI first
- [ ] Upload to PyPI: `twine upload dist/*`

## 📚 TODO After Release

### Documentation
- [ ] Set up ReadTheDocs
- [ ] Add API documentation
- [ ] Create tutorial notebooks
- [ ] Add more examples

### Community
- [ ] Announce on relevant forums
- [ ] Share on Twitter/social media
- [ ] Add to awesome-single-cell lists
- [ ] Consider writing blog post

### Maintenance
- [ ] Set up issue templates
- [ ] Set up pull request template
- [ ] Add code of conduct
- [ ] Monitor issues and PRs
- [ ] Plan next version features

## 🔍 Quick Verification Commands

```bash
# Check for Chinese characters
grep -r "[\u4e00-\u9fff]" src/scpd/*.py

# Test installation
pip install -e .

# Run tests
pytest tests/ -v

# Check package can be built
python -m build

# Verify package metadata
python -m twine check dist/*

# Test import
python -c "import scpd; print(scpd.__version__)"
```

## 📊 File Checklist

- [x] README.md - Enhanced and complete
- [x] LICENSE - MIT License added
- [x] CONTRIBUTING.md - Contribution guidelines
- [x] .gitignore - Python project gitignore
- [x] pyproject.toml - Updated metadata
- [x] .github/workflows/tests.yml - CI configuration
- [ ] CHANGELOG.md - Consider adding for releases
- [ ] CODE_OF_CONDUCT.md - Consider adding

## 🎯 URLs to Update

Replace `yourusername` with your actual GitHub username:

1. **README.md**:
   - Line ~8: Installation git clone URL
   - Line ~200: Citation URL
   - Line ~210: Support/Issues URL

2. **pyproject.toml**:
   - Homepage URL
   - Documentation URL
   - Repository URL
   - Bug Tracker URL
   - Changelog URL

3. **CONTRIBUTING.md**:
   - Line ~7: Clone URL

4. **LICENSE**:
   - Line 3: Copyright holder name

## ✨ Final Check

Before making your repository public:
- [ ] All placeholder text replaced
- [ ] No sensitive information in code
- [ ] No hardcoded paths or credentials
- [ ] All tests passing
- [ ] README looks good on GitHub preview
- [ ] License is appropriate
- [ ] Ready to accept contributions

---

**Note**: This checklist is comprehensive. Prioritize the "Critical" items before first commit, then work through others as you prepare for release.
