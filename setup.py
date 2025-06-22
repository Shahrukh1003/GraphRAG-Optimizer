from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="graphrag-optimizer",
    version="1.0.0",
    author="GraphRAG Optimizer Team",
    author_email="contact@graphrag-optimizer.com",
    description="Advanced GraphRAG implementation with hybrid retrieval and MCP architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/GraphRAG-Optimizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.8",
            "black>=21.0",
            "isort>=5.0",
            "mypy>=0.800",
            "bandit>=1.6",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "graphrag=advanced_graphrag:main",
            "graphrag-test=test_advanced_graphrag:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.yml", "*.yaml"],
    },
    keywords="graphrag, rag, retrieval, knowledge-graph, vector-database, ai, ml, nlp",
    project_urls={
        "Bug Reports": "https://github.com/your-username/GraphRAG-Optimizer/issues",
        "Source": "https://github.com/your-username/GraphRAG-Optimizer",
        "Documentation": "https://github.com/your-username/GraphRAG-Optimizer#readme",
    },
) 