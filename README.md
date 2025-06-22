# Advanced GraphRAG Optimizer

A production-ready, full Python implementation of GraphRAG (Graph Retrieval-Augmented Generation) combining vector databases, knowledge graphs, and advanced retrieval algorithms for explainable, anti-hallucination AI answers.

## 🚀 Features
- **Vector Similarity Search** (FAISS + SentenceTransformers)
- **Knowledge Graph Traversal** (Neo4j)
- **Hybrid Ranking** (vector + graph metrics)
- **MCP Architecture** (Model-Context-Prompt)
- **Graph Analytics** (centrality, connectivity)
- **Real-time Web UI** (Streamlit, for visualization only)
- **REST API** (FastAPI)
- **Automatic Knowledge Graph Building** from `data.txt`

## ✅ Minimum Requirements Checklist
| Requirement | Satisfied? | How/Where |
|-------------|:----------:|-----------|
| **Working RAG pipeline in Python** | ✅ | All retrieval, graph, and LLM logic in Python (`advanced_graphrag.py`, `graphrag_streamlit.py`). No low-code tools. |
| **Semantic retrieval + LLM response generation** | ✅ | FAISS + SentenceTransformers for semantic search; Neo4j for graph traversal; Ollama/Mistral for LLM answers. |
| **Published on GitHub** | ✅ | This repository is public and fully documented. |
| **Familiarity with embedding models** | ✅ | Uses SentenceTransformers for embeddings. |
| **Familiarity with vector search** | ✅ | Implements FAISS-based vector search. |
| **Familiarity with prompt engineering** | ✅ | Dynamic prompt construction and anti-hallucination rules in LLM pipeline. |

## 🛠 Technical Stack
- Python 3.8+
- FAISS, Neo4j, SentenceTransformers, PyTorch
- Ollama (Mistral LLM)
- FastAPI, Streamlit (for visualization only), NetworkX
- Docker, Uvicorn

## 📋 Requirements
- Python 3.8+
- Neo4j AuraDB account
- Ollama installed and running
- At least 4GB RAM

## 🚀 Installation
1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd GraphRAG-Optimizer
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set Up Environment Variables**
   Copy and edit `.env.example` as `.env`:
   ```env
   NEO4J_URI=neo4j+s://your-database-id.databases.neo4j.io:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your-password
   ```
4. **Install and Start Ollama**
   ```bash
   # Download from https://ollama.com/download
   ollama serve
   ollama pull mistral
   ```
5. **Prepare Data**
   Edit `data.txt` with your domain knowledge. The knowledge graph is auto-built from this file at startup.

## 🎯 Usage

### Web Interface (Visualization)
```bash
streamlit run graphrag_streamlit.py
```
- Ask questions and view real-time knowledge graph visualization.
- The graph is rebuilt from `data.txt` every time the app starts.
- **Note:** Streamlit is used only for visualization; all retrieval and LLM logic is in Python code.

### Advanced API (Production Backend)
```bash
python advanced_graphrag.py
```
- Starts a FastAPI server with REST endpoints for advanced retrieval and generation.
- See OpenAPI docs at `/docs` when running.

## 🔧 Architecture
- **Hybrid Retrieval:** Combines vector search and graph traversal for robust context.
- **Knowledge Graph:** Auto-generated from `data.txt` at startup (no manual Cypher needed).
- **MCP Pipeline:** Model (Mistral LLM via Ollama), Context (hybrid retrieval), Prompt (query classification).
- **Web UI:** Streamlit app for interactive Q&A and graph visualization.
- **API:** FastAPI backend for integration and automation.

## 📚 Example Data (data.txt)
```
Cats have retractable claws.
Cats can jump 5 times their height.
Tigers are the largest cat species.
Domestic cats sleep 12-16 hours daily.
```
Add your own facts to expand the system's knowledge.

## 🐳 Docker Deployment
```bash
docker build -t graphrag-optimizer .
docker run -p 8000:8000 graphrag-optimizer
```

## 📝 Contributing & Support
- See `CONTRIBUTING.md` for guidelines.
- For issues, use GitHub Issues.
- For questions, see the troubleshooting section or open a discussion.

## 📄 License
MIT License. See LICENSE for details.

---
**Built with ❤️ for advanced AI retrieval systems** 