# Contributing to scPD

Thank you for your interest in contributing to scPD! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/scpd.git
   cd scpd
   ```
3. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure:
   - Code follows existing style conventions
   - All tests pass: `pytest tests/`
   - New features include tests
   - Documentation is updated if needed

3. Commit your changes:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

4. Push to your fork and submit a pull request

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and modular

## Testing

Run tests with:
```bash
pytest tests/
```

Run tests with coverage:
```bash
pytest --cov=scpd tests/
```

## Documentation

- Update docstrings for any modified functions
- Update README.md if adding new features
- Add examples for new functionality

## Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Ensure all tests pass
- Keep pull requests focused on a single feature or fix

## Reporting Issues

When reporting issues, please include:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and relevant package versions
- Minimal code example if applicable

## Questions?

Feel free to open an issue for questions or discussions about the project.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
