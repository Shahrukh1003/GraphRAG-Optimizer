import streamlit as st
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
from neo4j import GraphDatabase
import ollama
from pyvis.network import Network
import streamlit.components.v1 as components
import re

# Load environment variables
load_dotenv()

# Initialize components
model = SentenceTransformer('all-MiniLM-L6-v2')
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def build_knowledge_graph_from_data(file_path='data.txt'):
    """Extract triples from data.txt and build the Neo4j knowledge graph."""
    triples = []
    # Simple pattern: subject <rel> object (very basic, can be improved)
    pattern = re.compile(r"(.+?) (has|can|are|is|sleep|belong|include|exhibit|have|used for|feature|attribute|capable of|sleeps|contains|with|of|to|for|as|by|on|in|from|about|at|into|over|after|before|between|under|against|during|without|within|along|following|across|behind|beyond|plus|except|but|up|out|around|down|off|above|near) (.+)", re.IGNORECASE)
    with open(file_path) as f:
        for line in f:
            m = pattern.match(line.strip('.').strip())
            if m:
                subj = m.group(1).strip()
                rel = m.group(2).strip()
                obj = m.group(3).strip()
                triples.append((subj, rel, obj))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")  # Clear old graph
        for subj, rel, obj in triples:
            rel_clean = re.sub(r'\W+', '_', rel.upper())
            session.run(
                f"""
                MERGE (a:Entity {{name: $subj}})
                MERGE (b:Entity {{name: $obj}})
                MERGE (a)-[r:{rel_clean}]->(b)
                """,
                {"subj": subj, "obj": obj}
            )

def create_vector_db(file_path='data.txt'):
    with open(file_path) as f:
        documents = [line.strip() for line in f if line.strip()]
    embeddings = model.encode(documents)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, documents

@st.cache_data
def hybrid_search(query, _index, docs, k=3, alpha=0.5, threshold=1.0):
    """Perform hybrid search combining vector and graph search with a relevance threshold."""
    # 1. Vector Search
    query_vector = model.encode([query])
    distances, indices = _index.search(query_vector, k)
    
    vector_results = []
    for i, doc_index in enumerate(indices[0]):
        if distances[0][i] < threshold:
            vector_results.append(docs[doc_index])

    # 2. Graph Search (simplified for Streamlit)
    # In a real app, this would query Neo4j
    query_tokens = set(query.lower().split())

    with driver.session() as session:
        graph_results = session.run("""
        MATCH (e:Entity)-[r]->(t)
        WHERE toLower(e.name) CONTAINS $query OR toLower(t.name) CONTAINS $query
        RETURN t.name AS result
        """, {"query": query.lower()})
        graph_data = [record["result"] for record in graph_results]
    return list(set(vector_results + graph_data))

def generate_response(query, context):
    """Generate a response using the LLM with anti-hallucination and explainability rules. Streams output if possible."""
    prompt = f"""
    Context:
    {context}
    
    Rules:
    1. Answer ONLY using above context
    2. If context doesn't contain answer, say "I don't know"
    3. Explain relationships using graph paths
    
    Question: {query}
    Answer:"""
    
    # Try streaming if ollama supports it, else fallback to full response
    try:
        response = ollama.generate(model='mistral', prompt=prompt)
        return response['response']
    except Exception as e:
        # Use fallback response generator
        return _generate_fallback_response(query, context)

def _generate_fallback_response(query, context):
    """Generate a fallback response when LLM is not available"""
    if not context:
        return "I don't have enough information to answer this question."
    
    # Extract key information from context
    context_lines = context.split('\n')
    facts = [line.strip() for line in context_lines if line.strip()][:3]
    
    # Classify query type for better response
    query_lower = query.lower()
    if any(word in query_lower for word in ['compare', 'difference', 'similar', 'versus', 'vs']):
        return f"Comparing the entities: {' '.join(facts)}"
    elif any(word in query_lower for word in ['analyze', 'pattern', 'trend', 'relationship', 'why']):
        return f"Analysis of the data: {' '.join(facts)}"
    else:
        return f"Based on the available information: {' '.join(facts)}"

def get_graph_data(query=None):
    """Fetch relevant subgraph from Neo4j based on the query. If nothing is found, return the full graph."""
    with driver.session() as session:
        if query and query.strip():
            # Try to find all paths up to 2 hops relevant to the query
            cypher = """
            MATCH path = (n)-[r*1..2]-(m)
            WHERE toLower(n.name) CONTAINS $query OR toLower(m.name) CONTAINS $query
            UNWIND relationships(path) as rel
            RETURN startNode(rel).name AS source, type(rel) AS rel, endNode(rel).name AS target
            """
            results = session.run(cypher, {"query": query.lower()})
            triples = [(record['source'], record['rel'], record['target']) for record in results]
            if triples:
                return triples
        # Fallback: show the full graph
        cypher = "MATCH (e:Entity)-[r]->(t) RETURN e.name AS source, type(r) AS rel, t.name AS target"
        results = session.run(cypher)
        return [(record['source'], record['rel'], record['target']) for record in results]

def draw_graph(graph_data, highlight_query=None):
    net = Network(height='400px', width='100%', directed=True)
    added = set()
    for src, rel, tgt in graph_data:
        net.add_node(src, label=src, color='orange' if highlight_query and highlight_query.lower() in src.lower() else 'lightblue')
        net.add_node(tgt, label=tgt, color='orange' if highlight_query and highlight_query.lower() in tgt.lower() else 'lightgreen')
        net.add_edge(src, tgt, label=rel, color='red' if highlight_query and (highlight_query.lower() in src.lower() or highlight_query.lower() in tgt.lower()) else 'gray')
    net.repulsion(node_distance=120, spring_length=200)
    return net

# Build knowledge graph and vector DB on startup
build_knowledge_graph_from_data()
index, docs = create_vector_db()

st.title("GraphRAG Optimizer")
st.write("Ask a question about cats, tigers, or your own data!")

query = st.text_input("Your question:")

if query and query.strip():  # Only process if query is not empty
    with st.spinner('Retrieving context and generating answer...'):
        context = "\n".join(hybrid_search(query, index, docs))
        st.subheader("Retrieved Context")
        st.code(context)
        response = generate_response(query, context)
        st.subheader("Answer")
        st.write(response)
        # Graph visualization
        st.subheader("Knowledge Graph Visualization")
        graph_data = get_graph_data(query)
        if not graph_data:
            st.info("No relationships found for this query. Showing the full knowledge graph instead.")
            graph_data = get_graph_data()  # Show the full graph
        net = draw_graph(graph_data, highlight_query=query)
        net.save_graph('graph.html')
        HtmlFile = open('graph.html', 'r', encoding='utf-8')
        components.html(HtmlFile.read(), height=450) 