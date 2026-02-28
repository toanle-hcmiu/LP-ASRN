# Contributing to LP-ASRN

Thank you for your interest in contributing to LP-ASRN! This document provides guidelines for contributing to the project.

## Documentation Requirements

**CRITICAL: When making code changes, you MUST update all affected documentation immediately.**

Documentation is not an afterthought - it is a essential part of the codebase. Any change that affects user-facing behavior, training procedures, or APIs must be reflected in the documentation.

### Documentation Update Checklist

Before submitting a pull request, verify:

- [ ] **README.md** reflects new features (for user-facing changes)
- [ ] **AGENTS.md** updated (for training stage or agent changes)
- [ ] **docs/architecture.md** updated (for model or loss changes)
- [ ] **docs/training.md** updated (for training-related changes)
- [ ] **Config file examples** updated (if parameters changed)
- [ ] **Docstrings updated** for modified functions/classes
- [ ] **docs/CHANGES.md** entry added (for bug fixes and features)
- [ ] **Pipeline tests pass**: `python scripts/test_pipeline.py`

### What to Document

| Change Type | Documentation to Update |
|-------------|-------------------------|
| New training stage | AGENTS.md, README.md, docs/training.md |
| New config parameter | README.md, docs/training.md, configs/lp_asrn.yaml |
| Model/architecture changes | AGENTS.md, docs/architecture.md, README.md |
| Loss function changes | docs/architecture.md, AGENTS.md |
| API changes | Docstrings in modified files |
| Bug fixes | docs/CHANGES.md |
| Inference changes | README.md, docs/training.md |
| Performance improvements | README.md, docs/CHANGES.md |

### Architecture Notes

- **Active generator**: RRDB-EA (in `src/models/generator.py`)
- **Backup generator**: SwinIR (in `src/models/generator_swinir_backup.py`)
- Inference auto-detects architecture from checkpoint keys
- When modifying the generator, update both `generator.py` and docs

## Code Standards

### Python Style

- Follow **PEP 8** style guidelines
- Use **type hints** for function signatures
- Add **docstrings** to all new functions and classes
- Maximum line length: 120 characters

### Docstring Format

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of what the function does.

    Longer description if needed. Explain the purpose, algorithm,
    or any important implementation details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: If param1 is invalid
    """
```

## Development Workflow

### 1. Branch Naming

```
feature/description-of-feature
fix/description-of-bug-fix
docs/update-documentation
```

### 2. Before Making Changes

1. Read the existing documentation
2. Understand the current architecture
3. Plan your changes
4. Identify which documentation needs updating

### 3. Making Changes

1. Write code following the style guide
2. Update documentation **as you code** (not after)
3. Add tests for new features
4. Run existing tests to ensure no regressions

### 4. Before Submitting

1. **Update documentation** - This is not optional
2. Run tests and ensure they pass
3. Format your code with appropriate tools
4. Write a clear pull request description

## Pull Request Guidelines

### PR Title Format

```
[type]: brief description

Examples:
feat: add support for new license plate layout
fix: correct tensor dimension mismatch in LCOFL loss
docs: update training guide with Stage 0 information
```

### PR Description Template

```markdown
## Summary
Brief description of what this PR does and why.

## Changes
- List of major changes

## Documentation Updates
- List of documentation files updated

## Testing
- How changes were tested

## Checklist
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests passing
```

## Testing

### Running Tests

```bash
# Run comprehensive pipeline tests (14 tests)
python scripts/test_pipeline.py

# Run unit tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_gradient_flow.py

# Run with coverage
python -m pytest --cov=src tests/
```

### Writing Tests

- Write tests for all new functionality
- Add pipeline tests to `scripts/test_pipeline.py` for integration coverage
- Use descriptive test names
- Include edge cases
- Test both RRDB-EA and SwinIR code paths when applicable

## Questions?

If you have questions about contributing, please open an issue or discussion.

---

Thank you for following these guidelines! By maintaining high standards for code and documentation quality, we ensure that LP-ASRN remains accessible and useful for everyone.
