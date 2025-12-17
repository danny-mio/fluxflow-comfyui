# Contributing to FluxFlow ComfyUI

Thank you for your interest in contributing to FluxFlow ComfyUI! This guide will help you get started with development.

## Quick Start for Developers

### 1. Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/danny-mio/fluxflow-comfyui.git
cd fluxflow-comfyui

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (automatic code quality checks)
pre-commit install
```text
### 2. Development Workflow

#### Before Making Changes

```bash
# Create a new branch
git checkout -b feature/your-feature-name
```text
#### During Development

```bash
# Format your code
black . --line-length 100
isort .

# Run linting
flake8

# Run type checking
mypy src/

# Run tests
pytest tests/ -v
```text
#### Before Committing

Pre-commit hooks will automatically run when you commit. To run them manually:

```bash
# Run all pre-commit checks
pre-commit run --all-files
```text
### 3. Code Quality Standards

FluxFlow ComfyUI uses several tools to maintain code quality:

- **Black**: Code formatting (line length: 100)
- **isort**: Import sorting (profile: black)
- **flake8**: Linting (max complexity: 15)
- **mypy**: Type checking
- **pytest**: Testing

For detailed code style guidelines, see [AGENTS.md](AGENTS.md).

## Code Style Guidelines

See [AGENTS.md](AGENTS.md) for comprehensive code style rules including:

- Python >= 3.10 with type hints on public APIs
- Google-style docstrings for all public functions/classes
- Naming conventions (snake_case, PascalCase, UPPER_SNAKE)
- Import ordering (stdlib → third-party → local)
- Error handling best practices
- Function length limits (< 50 lines)

### Example

```python
"""Module docstring explaining the module's purpose."""

from typing import Optional

from comfyui_fluxflow.utils import setup_logger

logger = setup_logger(__name__)


def process_data(input_path: str, output_path: Optional[str] = None) -> dict[str, int]:
    """
    Process data from input file and optionally save to output.

    Args:
        input_path: Path to input data file
        output_path: Optional path to save processed data

    Returns:
        Dictionary with processing statistics

    Raises:
        FileNotFoundError: If input_path doesn't exist
    """
    logger.info(f"Processing data from {input_path}")
    # Implementation here
    return {"processed": 42}
```text
### Documentation

- Update README.md for user-facing changes
- Keep documentation concise and practical
- Use code examples where helpful

## Testing

### Writing Tests

All new code should include tests:

```bash
# Create test file in appropriate directory
tests/unit/test_your_feature.py
tests/integration/test_your_workflow.py
```text
### Test Structure

```python
"""Tests for your feature."""

import pytest
from comfyui_fluxflow.your_module import YourClass


class TestYourClass:
    """Tests for YourClass."""

    def test_basic_functionality(self):
        """Test basic functionality works."""
        obj = YourClass()
        result = obj.method()
        assert result == expected

    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            obj = YourClass(invalid_param="bad")
```text
### Running Tests

```bash
# All tests
pytest tests/ -v

# Fast tests only (unit tests)
pytest tests/unit/ -v

# Specific test file
pytest tests/unit/test_your_feature.py -v

# Specific test function
pytest tests/unit/test_your_feature.py::test_function_name -v
```text
## Project Structure

```text
fluxflow-comfyui/
├── src/
│   └── comfyui_fluxflow/      # Main package
│       ├── __init__.py         # Node registration
│       ├── nodes/              # Node implementations
│       ├── schedulers.py       # Scheduler factory
│       ├── model_inspector.py  # Auto-detection
│       └── web/                # JavaScript extensions
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── pyproject.toml              # Project configuration
├── AGENTS.md                   # Code style reference
└── README.md                   # User documentation
```text
This is a standalone repository, independently installable via pip.

## Pull Request Process

1. **Create a branch** from `develop` or `main`
1. **Make your changes** with tests and documentation
1. **Run all checks**: `pre-commit run --all-files`
1. **Commit with clear messages**:
   ```bash
   git commit -m "Add feature: brief description

   Detailed explanation of what changed and why.

   - Specific change 1
   - Specific change 2"
   ```
1. **Push and create PR** on GitHub
1. **Address review feedback** if any

## Common Tasks

### Adding a New Feature

1. Create feature branch: `git checkout -b feature/name`
1. Add implementation in appropriate module
1. Add type hints and docstrings
1. Add tests for the feature
1. Update documentation if user-facing
1. Run `pre-commit run --all-files` to verify quality
1. Commit and create PR

### Fixing a Bug

1. Create bugfix branch: `git checkout -b fix/issue-description`
1. Add a test that reproduces the bug
1. Fix the bug
1. Verify the test passes
1. Run `pre-commit run --all-files`
1. Commit and create PR

### Adding Dependencies

1. Add to appropriate section in `pyproject.toml`:
   - Core dependencies → `dependencies`
   - Dev dependencies → `optional-dependencies.dev`
1. Document why the dependency is needed in your PR

## Requesting Contributor Access

### Public Contributors (No Access Needed)

Anyone can contribute via pull requests! Just:

1. Fork the repository
1. Make changes in your fork
1. Submit a pull request to the `develop` branch

No special permissions needed!

### Requesting Direct Repository Access

If you want to contribute regularly:

1. **Start by contributing** - Submit 1-3 quality PRs first
1. **Open a discussion** at https://github.com/danny-mio/fluxflow-comfyui/discussions
1. **Use title**: `Request: Contributor Access for [Your Name]`
1. **Include**: GitHub username, merged PRs, contribution areas, availability, experience

Repository owner reviews requests within 7 days.

## Security Issues

For security vulnerabilities, see [SECURITY.md](SECURITY.md).

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue with reproduction steps
- **Features**: Open a GitHub Issue to discuss before implementing

## Code of Conduct

- Be respectful and professional
- Focus on constructive feedback
- Welcome newcomers and help them learn
- Keep discussions focused on technical merit

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to FluxFlow ComfyUI!**
