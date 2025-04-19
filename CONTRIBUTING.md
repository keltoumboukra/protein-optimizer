# Contributing to Protein Expression Optimizer

This document provides guidelines and checklists for contributing to the Protein Expression Optimizer project.

## Development Workflow

1. **Setup Development Environment**
   ```bash
   # Clone the repository
   git clone https://github.com/keltoumboukra/protein-optimizer.git
   cd protein-optimizer

   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install development dependencies
   pip install -e ".[dev]"
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Checklist

### Before Starting Development
- [ ] Create a new branch from `main`
- [ ] Update your local `main` branch
- [ ] Install all development dependencies

### During Development
- [ ] Follow PEP 8 style guide
- [ ] Add type hints to all new functions
- [ ] Write docstrings for new functions/classes
- [ ] Write unit tests for new functionality
- [ ] Update documentation as needed

### Before Committing
Run these checks locally:
```bash
# Format code
black .

# Check types
mypy src/

# Run tests with coverage
pytest --cov=src tests/

# Check documentation coverage
interrogate --fail-under=80 src/ tests/
```

### CI Requirements
The following must pass in CI:
- [ ] Code formatting (Black)
- [ ] Type checking (mypy)
- [ ] Documentation coverage (interrogate)
- [ ] Unit tests
- [ ] Integration tests
- [ ] Dependency security (Safety)

### Code Quality Standards
- Maintain test coverage above 80%
- All new functions must have type hints
- All new functions must have docstrings
- All tests must pass
- No security vulnerabilities in dependencies

## Pull Request Process

1. **Before Submitting PR**
   - [ ] Run all local checks
   - [ ] Update documentation
   - [ ] Add tests for new features
   - [ ] Update README if needed

2. **PR Description**
   - Describe the changes
   - Link related issues
   - List any breaking changes
   - Update the changelog

3. **Review Process**
   - Address reviewer comments
   - Ensure CI passes
   - Keep PR focused and small

## Testing Guidelines

1. **Unit Tests**
   - Test each function independently
   - Mock external dependencies
   - Test edge cases
   - Aim for high coverage

2. **Integration Tests**
   - Test component interactions
   - Test API endpoints
   - Test data pipeline

## Documentation Guidelines

1. **Code Documentation**
   - Use clear, concise docstrings
   - Include type hints
   - Document exceptions
   - Add examples for complex functions

2. **Project Documentation**
   - Update README.md for new features
   - Keep API documentation current
   - Document breaking changes

## Getting Help

- Open an issue for bugs
- Use discussions for questions
- Join our community chat

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License. 