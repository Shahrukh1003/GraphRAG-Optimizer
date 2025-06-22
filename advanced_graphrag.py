import os
import json
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss
from neo4j import GraphDatabase
import ollama
from dotenv import load_dotenv
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class RetrievalResult:
    """Data class for retrieval results with ranking scores"""
    content: str
    source: str  # 'vector' or 'graph'
    similarity_score: float
    graph_centrality: float
    entity_connectivity: int
    final_score: float

class GraphRAGAdvanced:
    """Advanced GraphRAG implementation with hybrid retrieval and MCP architecture"""
    
    def __init__(self):
        # Initialize components
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Try to connect to Neo4j, fall back to demo mode
        try:
            self.driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI"),
                auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
            )
            self.demo_mode = False
        except Exception as e:
            logger.warning(f"Neo4j not available, using demo mode: {e}")
            self.driver = None
            self.demo_mode = True
        
        # Vector database
        self.vector_index = None
        self.documents = []
        
        # Graph metrics cache
        self.graph_metrics = {}
        
        # Demo graph data for when Neo4j is not available
        self.demo_graph_data = [
            ("Cats", "HAS_ATTRIBUTE", "retractable claws"),
            ("Cats", "CAPABLE_OF", "jump 5x height"),
            ("Tigers", "IS_A", "largest cat"),
            ("Cats", "EXHIBITS", "sleep 12-16 hours"),
            ("Tigers", "BELONGS_TO", "Felidae"),
            ("Cats", "BELONGS_TO", "Felidae"),
            ("Domestic Cats", "IS_A", "Cats"),
            ("Wild Cats", "INCLUDES", "Tigers")
        ]
        
        # Prompt templates for different query types
        self.prompt_templates = {
            'factual': """
            Context: {context}
            
            Instructions:
            1. Answer ONLY using the provided context
            2. If the answer is not in the context, say "I don't have enough information"
            3. Cite specific facts from the context
            4. Explain relationships between entities when relevant
            
            Question: {query}
            Answer:""",
            
            'analytical': """
            Context: {context}
            
            Instructions:
            1. Analyze the relationships and patterns in the context
            2. Provide insights about entity connections
            3. Identify any trends or correlations
            4. If insufficient data, state what additional information would help
            
            Question: {query}
            Analysis:""",
            
            'comparative': """
            Context: {context}
            
            Instructions:
            1. Compare entities based on the provided context
            2. Highlight similarities and differences
            3. Use graph relationships to explain comparisons
            4. If comparison is not possible, explain why
            
            Question: {query}
            Comparison:"""
        }
    
    def create_vector_database(self, file_path: str = 'data.txt') -> None:
        """Create FAISS vector index with advanced indexing"""
        logger.info("Creating vector database...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.documents = [line.strip() for line in f if line.strip()]
        
        # Generate embeddings
        embeddings = self.model.encode(self.documents, show_progress_bar=True)
        
        # Create FAISS index with L2 distance
        dimension = embeddings.shape[1]
        self.vector_index = faiss.IndexFlatL2(dimension)
        self.vector_index.add(embeddings.astype('float32'))
        
        logger.info(f"Vector database created with {len(self.documents)} documents")
    
    def extract_triples_from_text(self, file_path: str = 'data.txt') -> list:
        """Extract (subject, relation, object) triples from each line in data.txt using simple patterns."""
        triples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Simple pattern: "X [verb] Y" or "X is Y"
                m = re.match(r'(.+?) (is|are|has|have|can|controls|powers|slows|distributes|receives|generates|applies|stops|turns|controls|contains|belongs to|includes|exhibits|sleeps|jumps|eats|drinks|lives|runs|flies|boils at|freezes at|capital of|president of|.+?) (.+)', line, re.IGNORECASE)
                if m:
                    subj = m.group(1).strip('.').strip()
                    rel = m.group(2).strip('.').strip()
                    obj = m.group(3).strip('.').strip()
                    triples.append((subj, rel, obj))
        return triples

    def build_knowledge_graph(self) -> None:
        """Build knowledge graph from auto-generated triples."""
        triples = self.extract_triples_from_text()
        if self.demo_mode:
            logger.info("Building demo knowledge graph (in-memory) from data.txt")
            self.demo_graph_data = triples
            return
        logger.info("Building knowledge graph in Neo4j from data.txt...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            for subj, rel, obj in triples:
                # Use MERGE to avoid duplicates
                session.run(
                    """
                    MERGE (a:Entity {name: $subj})
                    MERGE (b:Entity {name: $obj})
                    MERGE (a)-[r:%s]->(b)
                    """ % re.sub(r'\W+', '_', rel.upper()),
                    {"subj": subj, "obj": obj}
                )
        logger.info("Knowledge graph built successfully from data.txt")
    
    def calculate_graph_metrics(self) -> Dict:
        """Calculate graph centrality and connectivity metrics from current graph data."""
        if self.demo_mode:
            logger.info("Calculating demo graph metrics from data.txt...")
            G = nx.DiGraph()
            for src, rel, tgt in self.demo_graph_data:
                G.add_edge(src, tgt, relation=rel)
            
            # Calculate metrics
            metrics = {}
            for node in G.nodes():
                metrics[node] = {
                    'degree_centrality': nx.degree_centrality(G).get(node, 0),
                    'betweenness_centrality': nx.betweenness_centrality(G).get(node, 0),
                    'closeness_centrality': nx.closeness_centrality(G).get(node, 0),
                    'in_degree': G.in_degree(node),
                    'out_degree': G.out_degree(node),
                    'total_degree': G.degree(node)
                }
            
            self.graph_metrics = metrics
            logger.info(f"Demo graph metrics calculated for {len(metrics)} nodes")
            return metrics
        
        logger.info("Calculating graph metrics...")
        
        with self.driver.session() as session:
            # Get all nodes and relationships
            result = session.run("""
            MATCH (n)-[r]->(m)
            RETURN n.name as source, type(r) as rel, m.name as target
            """)
            
            # Build NetworkX graph for analysis
            G = nx.DiGraph()
            for record in result:
                G.add_edge(record['source'], record['target'], relation=record['rel'])
            
            # Calculate metrics
            metrics = {}
            for node in G.nodes():
                metrics[node] = {
                    'degree_centrality': nx.degree_centrality(G).get(node, 0),
                    'betweenness_centrality': nx.betweenness_centrality(G).get(node, 0),
                    'closeness_centrality': nx.closeness_centrality(G).get(node, 0),
                    'in_degree': G.in_degree(node),
                    'out_degree': G.out_degree(node),
                    'total_degree': G.degree(node)
                }
            
            self.graph_metrics = metrics
            logger.info(f"Graph metrics calculated for {len(metrics)} nodes")
            return metrics
    
    def vector_similarity_search(self, query: str, k: int = 3, threshold: float = 1.0) -> List[str]:
        """Perform vector similarity search with a distance threshold."""
        if self.vector_index is None:
            logging.warning("Vector index not created. Returning empty list.")
            return []
        
        logging.info(f"Performing vector similarity search for query: {query}")
        query_vector = self.model.encode([query])
        distances, indices = self.vector_index.search(query_vector, k)
        
        results = []
        for i, doc_index in enumerate(indices[0]):
            if distances[0][i] < threshold:
                results.append(self.documents[doc_index])
            else:
                logging.info(f"Result {i} discarded due to high distance: {distances[0][i]}")
        
        return results
    
    def graph_traversal_search(self, query: str, max_depth: int = 2) -> List[RetrievalResult]:
        """Execute graph traversal algorithms to fetch contextual data"""
        logger.info(f"Performing graph traversal for query: {query}")
        
        if self.demo_mode:
            # Use demo graph data for traversal
            results = []
            query_lower = query.lower()
            
            for src, rel, tgt in self.demo_graph_data:
                if (query_lower in src.lower() or 
                    query_lower in tgt.lower() or 
                    query_lower in rel.lower()):
                    
                    content = f"{src} -> {rel} -> {tgt}"
                    
                    # Calculate entity connectivity
                    source_connectivity = self.graph_metrics.get(src, {}).get('total_degree', 0)
                    target_connectivity = self.graph_metrics.get(tgt, {}).get('total_degree', 0)
                    avg_connectivity = (source_connectivity + target_connectivity) / 2
                    
                    # Calculate centrality score
                    source_centrality = self.graph_metrics.get(src, {}).get('degree_centrality', 0)
                    target_centrality = self.graph_metrics.get(tgt, {}).get('degree_centrality', 0)
                    avg_centrality = (source_centrality + target_centrality) / 2
                    
                    final_score = avg_centrality
                    
                    results.append(RetrievalResult(
                        content=content,
                        source='graph',
                        similarity_score=0.0,
                        graph_centrality=avg_centrality,
                        entity_connectivity=avg_connectivity,
                        final_score=final_score
                    ))
            
            return results
        
        with self.driver.session() as session:
            # Multi-hop traversal with relevance scoring
            result = session.run("""
            MATCH path = (start:Entity)-[r*1..3]->(end)
            WHERE toLower(start.name) CONTAINS $query 
               OR toLower(end.name) CONTAINS $query
               OR toLower(type(r[0])) CONTAINS $query
            RETURN start.name as source, 
                   [rel in relationships(path) | type(rel)] as relationships,
                   end.name as target,
                   length(path) as path_length
            ORDER BY path_length ASC
            LIMIT 10
            """, {"query": query.lower()})
            
            results = []
            for record in result:
                content = f"{record['source']} -> {record['relationships']} -> {record['target']}"
                
                # Calculate entity connectivity
                source_connectivity = self.graph_metrics.get(record['source'], {}).get('total_degree', 0)
                target_connectivity = self.graph_metrics.get(record['target'], {}).get('total_degree', 0)
                avg_connectivity = (source_connectivity + target_connectivity) / 2
                
                # Calculate centrality score
                source_centrality = self.graph_metrics.get(record['source'], {}).get('degree_centrality', 0)
                target_centrality = self.graph_metrics.get(record['target'], {}).get('degree_centrality', 0)
                avg_centrality = (source_centrality + target_centrality) / 2
                
                # Path length penalty
                path_penalty = 1 / (record['path_length'] + 1)
                
                final_score = avg_centrality * path_penalty
                
                results.append(RetrievalResult(
                    content=content,
                    source='graph',
                    similarity_score=0.0,
                    graph_centrality=avg_centrality,
                    entity_connectivity=avg_connectivity,
                    final_score=final_score
                ))
            
            return results
    
    def hybrid_ranking(self, vector_results: List[RetrievalResult], 
                      graph_results: List[RetrievalResult], 
                      alpha: float = 0.6) -> List[RetrievalResult]:
        """Implement top-k ranking based on entity connectivity and graph metrics"""
        logger.info("Performing hybrid ranking...")
        
        all_results = vector_results + graph_results
        
        # Normalize scores
        max_sim = max([r.similarity_score for r in all_results]) if any(r.similarity_score > 0 for r in all_results) else 1
        max_centrality = max([r.graph_centrality for r in all_results]) if any(r.graph_centrality > 0 for r in all_results) else 1
        max_connectivity = max([r.entity_connectivity for r in all_results]) if any(r.entity_connectivity > 0 for r in all_results) else 1
        
        for result in all_results:
            # Normalize individual scores
            norm_sim = result.similarity_score / max_sim if max_sim > 0 else 0
            norm_centrality = result.graph_centrality / max_centrality if max_centrality > 0 else 0
            norm_connectivity = result.entity_connectivity / max_connectivity if max_connectivity > 0 else 0
            
            # Calculate final score with weighted combination
            result.final_score = (
                alpha * norm_sim + 
                (1 - alpha) * 0.5 * (norm_centrality + norm_connectivity)
            )
        
        # Sort by final score and return top-k
        ranked_results = sorted(all_results, key=lambda x: x.final_score, reverse=True)
        return ranked_results[:10]  # Return top 10
    
    def classify_query_type(self, query: str) -> str:
        """Classify query type for prompt tuning"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['compare', 'difference', 'similar', 'versus', 'vs']):
            return 'comparative'
        elif any(word in query_lower for word in ['analyze', 'pattern', 'trend', 'relationship', 'why']):
            return 'analytical'
        else:
            return 'factual'
    
    def generate_mcp_response(self, query: str, context_results: List[RetrievalResult]) -> str:
        """Generate response using MCP (Model-Context-Prompt) architecture"""
        logger.info("Generating MCP response...")
        
        # Prepare context
        context = "\n".join([f"- {result.content} (Score: {result.final_score:.3f})" 
                            for result in context_results])
        
        # Classify query type and select prompt template
        query_type = self.classify_query_type(query)
        prompt_template = self.prompt_templates[query_type]
        
        # Format prompt with context
        prompt = prompt_template.format(context=context, query=query)
        
        try:
            # Generate response using Ollama
            response = ollama.generate(model='mistral', prompt=prompt)
            return response['response']
        except Exception as e:
            logger.warning(f"Ollama not available, using fallback response generator: {e}")
            return self._generate_fallback_response(query, context_results, query_type)
    
    def _generate_fallback_response(self, query: str, context_results: List[RetrievalResult], query_type: str) -> str:
        """Generate a fallback response when LLM is not available"""
        if not context_results:
            return "I don't have enough information to answer this question."
        
        # Extract key information from context
        facts = [result.content for result in context_results[:3]]
        
        if query_type == 'factual':
            return f"Based on the available information: {' '.join(facts)}"
        elif query_type == 'comparative':
            return f"Comparing the entities: {' '.join(facts)}"
        elif query_type == 'analytical':
            return f"Analysis of the data: {' '.join(facts)}"
        else:
            return f"Here's what I found: {' '.join(facts)}"
    
    def retrieve_and_generate(self, query: str, k: int = 5) -> Dict:
        """Main retrieval and generation pipeline"""
        logger.info(f"Processing query: {query}")
        
        # Step 1: Vector similarity search
        vector_results = self.vector_similarity_search(query, k)
        
        # Step 2: Graph traversal search
        graph_results = self.graph_traversal_search(query)
        
        # Step 3: Hybrid ranking
        ranked_results = self.hybrid_ranking(vector_results, graph_results)
        
        # Step 4: Generate response using MCP architecture
        response = self.generate_mcp_response(query, ranked_results)
        
        return {
            'query': query,
            'vector_results': vector_results,
            'graph_results': graph_results,
            'ranked_results': ranked_results,
            'response': response,
            'query_type': self.classify_query_type(query)
        }

# FastAPI application for deployment
app = FastAPI(title="Advanced GraphRAG API", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    k: int = 5

class QueryResponse(BaseModel):
    query: str
    response: str
    vector_results: List[Dict]
    graph_results: List[Dict]
    ranked_results: List[Dict]
    query_type: str

# Initialize GraphRAG system
graphrag = GraphRAGAdvanced()

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    try:
        # Check if Neo4j is available
        try:
            graphrag.create_vector_database()
            graphrag.build_knowledge_graph()
            graphrag.calculate_graph_metrics()
            logger.info("GraphRAG system initialized successfully with Neo4j")
        except Exception as e:
            logger.warning(f"Neo4j not available, using demo mode: {e}")
            # Create demo data without Neo4j
            graphrag.create_vector_database()
            logger.info("GraphRAG system initialized in demo mode (vector-only)")
        
        # Check if Ollama is available
        try:
            ollama.list()
            logger.info("Ollama is available")
        except Exception as e:
            logger.warning(f"Ollama not available, will use fallback responses: {e}")
            
    except Exception as e:
        logger.error(f"Error initializing GraphRAG system: {e}")
        # Don't raise - allow the app to start in limited mode

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query using the advanced GraphRAG system"""
    try:
        result = graphrag.retrieve_and_generate(request.query, request.k)
        
        # Convert RetrievalResult objects to dictionaries for JSON serialization
        return QueryResponse(
            query=result['query'],
            response=result['response'],
            vector_results=[{
                'content': r.content,
                'source': r.source,
                'similarity_score': r.similarity_score,
                'final_score': r.final_score
            } for r in result['vector_results']],
            graph_results=[{
                'content': r.content,
                'source': r.source,
                'graph_centrality': r.graph_centrality,
                'entity_connectivity': r.entity_connectivity,
                'final_score': r.final_score
            } for r in result['graph_results']],
            ranked_results=[{
                'content': r.content,
                'source': r.source,
                'final_score': r.final_score
            } for r in result['ranked_results']],
            query_type=result['query_type']
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "system": "Advanced GraphRAG"}

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)

def main():
    """Main function for running the GraphRAG system directly"""
    try:
        # Initialize the system
        graphrag = GraphRAGAdvanced()
        graphrag.create_vector_database()
        graphrag.build_knowledge_graph()
        graphrag.calculate_graph_metrics()
        
        print("🚀 GraphRAG Optimizer is ready!")
        print("=" * 50)
        
        # Interactive mode
        while True:
            query = input("\nAsk a question (type 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            
            print("\n🔍 Processing query...")
            result = graphrag.retrieve_and_generate(query, k=5)
            
            print(f"\n📊 Query Type: {result['query_type']}")
            print(f"🤖 Response: {result['response']}")
            
            print(f"\n📈 Top Results:")
            for i, r in enumerate(result['ranked_results'][:3], 1):
                print(f"  {i}. {r.content} (Score: {r.final_score:.3f})")
            
            print("-" * 50)
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main() 