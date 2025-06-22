# Contributing to GraphRAG Optimizer

Thank you for your interest in contributing to the GraphRAG Optimizer project! This project is a full Python implementation of a modern RAG pipeline—no low-code tools are used for retrieval, graph, or LLM logic. Streamlit is used only for visualization.

## 🤝 How to Contribute

### 1. Fork and Clone
1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/GraphRAG-Optimizer.git
   cd GraphRAG-Optimizer
   ```

### 2. Set Up Development Environment
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your Neo4j credentials
   ```

### 3. Make Your Changes
1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the coding standards below

3. If you are updating the knowledge graph, edit `data.txt` (the graph is auto-built from this file at startup).

4. Add or update tests if you are modifying core logic (see advanced_graphrag.py for testable functions).

### 4. Submit Your Changes
1. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

2. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Create a Pull Request on GitHub

## 📋 Coding Standards

### Python Code Style
- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all functions and classes
- Keep functions focused and under 50 lines when possible

### Example Code Style
```python
from typing import List, Dict, Optional

def process_query(query: str, context: List[str]) -> Dict[str, any]:
    """
    Process a query using the GraphRAG system.
    
    Args:
        query: The user's question
        context: List of relevant context documents
        
    Returns:
        Dictionary containing the response and metadata
    """
    # Implementation here
    pass
```

### Commit Message Format
Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for adding tests
- `refactor:` for code refactoring
- `style:` for formatting changes

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python test_advanced_graphrag.py

# Run specific test functions
python -m pytest test_advanced_graphrag.py::test_vector_similarity_search
```

### Writing Tests
- Write tests for all new functionality
- Use descriptive test names
- Include both positive and negative test cases
- Mock external dependencies when appropriate

## 📚 Documentation

### Code Documentation
- All public functions must have docstrings
- Include examples in docstrings for complex functions
- Document any non-obvious algorithms or approaches

### README Updates
- Update README.md for new features
- Include usage examples
- Update installation instructions if needed

## 🚀 Development Workflow

### Before Submitting
1. Run all tests: `python test_advanced_graphrag.py`
2. Check code style: `flake8 .`
3. Update documentation
4. Test your changes locally

### Pull Request Guidelines
1. **Title**: Clear, descriptive title
2. **Description**: Explain what and why, not how
3. **Tests**: Include tests for new functionality
4. **Documentation**: Update relevant docs
5. **Screenshots**: Include for UI changes

## 🐛 Bug Reports

### Before Reporting
1. Check existing issues
2. Try the latest version
3. Reproduce the issue

### Bug Report Template
```markdown
**Bug Description**
Brief description of the issue

**Steps to Reproduce**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Windows 10, macOS 12]
- Python: [e.g., 3.9.7]
- Dependencies: [list relevant versions]

**Additional Context**
Any other relevant information
```

## 💡 Feature Requests

### Feature Request Template
```markdown
**Feature Description**
Brief description of the feature

**Use Case**
Why this feature would be useful

**Proposed Implementation**
Optional: how you think it could be implemented

**Alternatives Considered**
Other approaches you've considered
```

## �� Getting Help

- **Issues**: Use GitHub Issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Code Review**: All PRs will be reviewed by maintainers

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

This project demonstrates embedding models, vector search, and prompt engineering in a modern RAG pipeline. Thank you for contributing to GraphRAG Optimizer! 🚀 