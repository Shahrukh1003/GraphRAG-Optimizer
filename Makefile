.PHONY: help install test lint clean build docker-build docker-run docker-stop docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting checks"
	@echo "  clean       - Clean up temporary files"
	@echo "  build       - Build the package"
	@echo "  docker-build- Build Docker image"
	@echo "  docker-run  - Run with Docker Compose"
	@echo "  docker-stop - Stop Docker containers"
	@echo "  docs        - Generate documentation"

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e .

# Install development dependencies
install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Run tests
test:
	python test_advanced_graphrag.py

# Run tests with coverage
test-cov:
	pytest --cov=. --cov-report=html --cov-report=term

# Run linting
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Format code
format:
	black .
	isort .

# Type checking
type-check:
	mypy .

# Security check
security:
	bandit -r .

# Clean up
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

# Build package
build:
	python setup.py sdist bdist_wheel

# Docker commands
docker-build:
	docker build -t graphrag-optimizer .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Generate documentation
docs:
	sphinx-build -b html docs/ docs/_build/html

# Start development server
dev:
	streamlit run graphrag_streamlit.py

# Start production server
prod:
	uvicorn advanced_graphrag:app --host 0.0.0.0 --port 8000 --reload

# Quick setup for new developers
setup:
	@echo "Setting up GraphRAG Optimizer development environment..."
	python -m venv venv
	@echo "Virtual environment created. Activate it with:"
	@echo "  source venv/bin/activate  # On Unix/MacOS"
	@echo "  venv\\Scripts\\activate     # On Windows"
	@echo "Then run: make install-dev"

# Full development workflow
dev-workflow: install-dev lint test

# Production deployment check
prod-check: lint test security type-check

# Update dependencies
update-deps:
	pip install --upgrade pip
	pip install --upgrade -r requirements.txt

# Create release
release:
	@echo "Creating release..."
	@read -p "Enter version number (e.g., 1.0.1): " version; \
	git tag -a v$$version -m "Release version $$version"; \
	git push origin v$$version; \
	echo "Release v$$version created and pushed" 