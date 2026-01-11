# Contributing to LineamentLearning

Thank you for your interest in contributing to LineamentLearning! This document provides guidelines and instructions for contributing.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow:
- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Respect differing viewpoints

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/LineamentLearning.git
   cd LineamentLearning
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/RichardScottOZ/LineamentLearning.git
   ```

## Development Setup

### Install Development Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev,full,modern-ui]"

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Development Tools

We use several tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pytest**: Testing

Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

## Making Changes

### 1. Create a Branch

```bash
git checkout -b feature/my-new-feature
# or
git checkout -b fix/issue-123
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Adding tests

### 2. Make Your Changes

- Write clean, readable code
- Follow the style guide (see below)
- Add type hints to new functions
- Update documentation as needed
- Add tests for new functionality

### 3. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add U-Net architecture with attention mechanism

- Implement spatial and channel attention
- Add configuration options for attention
- Update documentation
- Add unit tests

Fixes #123"
```

Commit message format:
- First line: Short summary (50 chars or less)
- Blank line
- Detailed description (wrap at 72 chars)
- Reference issues and PRs

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_model.py

# Run specific test
pytest tests/test_model.py::test_build_unet
```

### Writing Tests

Example test structure:

```python
import pytest
from config import Config
from model_modern import build_model

class TestModelBuilding:
    """Test model building functionality."""
    
    def test_build_rotatenet(self):
        """Test RotateNet architecture creation."""
        config = Config()
        config.model.architecture = 'RotateNet'
        model = build_model(config)
        
        assert model is not None
        assert model.input_shape[1:] == (45, 45, 8)
        assert model.output_shape[1] == 1
    
    def test_build_unet(self):
        """Test U-Net architecture creation."""
        config = Config()
        config.model.architecture = 'UNet'
        model = build_model(config)
        
        assert model is not None
        assert model.name == 'UNet'
```

### Test Coverage

Aim for:
- **90%+ coverage** for new code
- **100% coverage** for critical paths
- Tests for edge cases and error conditions

## Pull Request Process

### Before Submitting

1. **Update your branch** with upstream:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests**:
   ```bash
   pytest
   ```

3. **Check code quality**:
   ```bash
   black .
   flake8 .
   mypy .
   ```

4. **Update documentation**:
   - Update README.md if needed
   - Update CHANGELOG.md
   - Add docstrings to new functions

### Submitting Pull Request

1. **Push to your fork**:
   ```bash
   git push origin feature/my-new-feature
   ```

2. **Create Pull Request** on GitHub

3. **Fill out PR template**:
   - Clear description of changes
   - Link to related issues
   - Screenshots for UI changes
   - Test results

4. **Request review** from maintainers

### PR Checklist

- [ ] Code follows style guide
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No merge conflicts
- [ ] CI/CD checks passing

## Style Guide

### Python Code Style

Follow **PEP 8** with these specifics:

#### Imports
```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Local
from config import Config
from model_modern import build_model
```

#### Type Hints
```python
from typing import List, Optional, Tuple

def train_model(
    config: Config,
    data_path: str,
    epochs: Optional[int] = None
) -> keras.Model:
    """Train a model with given configuration.
    
    Args:
        config: Configuration object
        data_path: Path to training data
        epochs: Number of epochs (uses config if None)
        
    Returns:
        Trained Keras model
    """
    pass
```

#### Docstrings
Use Google style:

```python
def function_with_docstring(param1: int, param2: str) -> bool:
    """Short description.
    
    Longer description if needed. Can span multiple lines
    and include examples.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is negative
        
    Example:
        >>> function_with_docstring(5, "test")
        True
    """
    pass
```

#### Naming Conventions

- **Classes**: `PascalCase` (e.g., `ModelTrainer`)
- **Functions/Methods**: `snake_case` (e.g., `build_model`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_EPOCHS`)
- **Private**: `_leading_underscore` (e.g., `_internal_method`)

#### Line Length

- Maximum 88 characters (Black default)
- Maximum 72 for docstrings/comments

#### Code Organization

```python
# 1. Module docstring
"""Module for model training."""

# 2. Imports
import tensorflow as tf

# 3. Constants
MAX_EPOCHS = 100

# 4. Classes and functions
class ModelTrainer:
    pass

def train_model():
    pass

# 5. Main execution guard
if __name__ == '__main__':
    main()
```

### Configuration Files

Use consistent formatting:

```json
{
    "model": {
        "architecture": "UNet",
        "window_size": 64
    }
}
```

### Documentation

- Use Markdown for documentation files
- Keep lines under 80 characters
- Use code blocks with language tags
- Include examples where helpful

## Areas for Contribution

### High Priority
- [ ] Add comprehensive test suite
- [ ] Create Jupyter notebook examples
- [ ] Implement Gradio/Streamlit dashboard
- [ ] Add data loading pipeline
- [ ] Docker containerization

### Medium Priority
- [ ] Add more model architectures
- [ ] Implement data augmentation pipeline
- [ ] Add model export (ONNX, TFLite)
- [ ] Create API server
- [ ] Add visualization tools

### Good First Issues
- [ ] Improve documentation
- [ ] Add type hints to legacy code
- [ ] Write unit tests
- [ ] Fix small bugs
- [ ] Add examples

## Questions?

- **Open an issue** for bugs or feature requests
- **Start a discussion** for questions or ideas
- **Contact maintainers** via GitHub

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

## Recognition

Contributors will be recognized in:
- README.md Contributors section
- Release notes
- CHANGELOG.md

Thank you for contributing to LineamentLearning! 🎉
