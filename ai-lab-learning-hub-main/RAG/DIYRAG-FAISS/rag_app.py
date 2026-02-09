import os
import json
import streamlit as st
import faiss
import numpy as np
import openai
from dotenv import load_dotenv
import logging
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_streamlit.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    "EMBEDDING_MODEL": "text-embedding-ada-002",
    "LLM_MODEL": "gpt-4.1-nano",
    "INDEX_PATH": "faiss.index",
    "METADATA_PATH": "metadata_store.json",
    "SIMILARITY_THRESHOLD": 0.7,
    "MAX_RESULTS": 5,
    "DATA_DIR": "./data/annual_reports"
}

# Sample queries for the interface
SAMPLE_QUERIES = [
    "What were the key financial metrics in the most recent annual reports?",
    "What were the main challenges mentioned in the annual reports?",
    "What were the strategic priorities for the companies?",
    "What were the key risks identified in the annual reports?",
    "What were the key achievements in 2023?",
    "What is the strategy for future growth?",
    "What were the R&D priorities?",
    "What were the financial results for the year?"
]

# Function to check if files exist
def check_files_exist():
    """Check if FAISS index and metadata files exist."""
    index_exists = os.path.exists(CONFIG["INDEX_PATH"])
    metadata_exists = os.path.exists(CONFIG["METADATA_PATH"])
    return index_exists and metadata_exists

# Function to save API key to .env file
def save_api_key_to_env(api_key):
    """Save API key to .env file."""
    try:
        with open(".env", "w") as f:
            f.write(f"OPENAI_API_KEY={api_key}")
        return True
    except Exception as e:
        logger.error(f"Error saving API key: {e}")
        return False

# Function to run the document processing script
def run_document_processing():
    """Run the document processing script."""
    try:
        # Check if data directory exists
        if not os.path.exists(CONFIG["DATA_DIR"]):
            os.makedirs(CONFIG["DATA_DIR"])
            st.info(f"Created data directory at {CONFIG['DATA_DIR']}. Please add your PDF files there.")
            return False
        
        # Check if there are PDF files in the data directory
        pdf_files = [f for f in os.listdir(CONFIG["DATA_DIR"]) if f.endswith('.pdf')]
        if not pdf_files:
            st.warning(f"No PDF files found in {CONFIG['DATA_DIR']}. Please add your PDF files there.")
            return False
        
        # Check if API key is set
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OpenAI API key not found. Please enter your API key in the sidebar first.")
            return False
        
        # Run the document processing script
        st.info("Running document processing script. This may take a while...")
        st.info("The script will use the OpenAI API key you entered in the sidebar.")
        
        result = subprocess.run([sys.executable, "load_chunk_embed_store.py"], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            st.success("Document processing completed successfully!")
            return True
        else:
            st.error(f"Error running document processing script: {result.stderr}")
            return False
    except Exception as e:
        st.error(f"Error running document processing script: {e}")
        return False

@st.cache_resource
def load_faiss_index_and_metadata():
    """Load the FAISS index and metadata from disk."""
    try:
        # Load FAISS index
        faiss_index = faiss.read_index(CONFIG["INDEX_PATH"])
        
        # Load metadata
        with open(CONFIG["METADATA_PATH"], 'r') as f:
            metadata_store = json.load(f)
        
        return faiss_index, metadata_store
    except Exception as e:
        st.error(f"Error loading FAISS index and metadata: {e}")
        return None, None

def get_embedding(text):
    """Generate an embedding for a single text via OpenAI API."""
    try:
        response = openai.embeddings.create(
            model=CONFIG["EMBEDDING_MODEL"],
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

def get_metadata_summary(metadata_store):
    """Get a summary of the metadata store."""
    if not metadata_store:
        return "No metadata available"
    
    # Get unique companies and their document counts
    companies = {}
    for item in metadata_store:
        company = item.get('company', 'Unknown')
        if company not in companies:
            companies[company] = 1
        else:
            companies[company] += 1
    
    return companies

def analyze_query_type(query):
    """Analyze the query to determine if it's broad, specific, or technical."""
    # Keywords that suggest broad queries
    broad_keywords = ['overview', 'summary', 'general', 'all', 'everything', 'comprehensive']
    # Keywords that suggest specific queries
    specific_keywords = ['specific', 'exact', 'particular', 'certain', 'precise']
    # Keywords that suggest technical queries
    technical_keywords = ['technical', 'detailed', 'specific number', 'exact value', 'precise data']
    
    query_lower = query.lower()
    
    # Count matches for each type
    broad_count = sum(1 for word in broad_keywords if word in query_lower)
    specific_count = sum(1 for word in specific_keywords if word in query_lower)
    technical_count = sum(1 for word in technical_keywords if word in query_lower)
    
    # Determine query type
    if broad_count > specific_count and broad_count > technical_count:
        return 'broad'
    elif technical_count > broad_count and technical_count > specific_count:
        return 'technical'
    else:
        return 'specific'

def get_chunk_count(query_type):
    """Get the appropriate number of chunks based on query type."""
    chunk_counts = {
        'broad': 15,    # More chunks for broad queries
        'specific': 7,  # Medium chunks for specific queries
        'technical': 5  # Fewer chunks for technical queries
    }
    return chunk_counts.get(query_type, 7)  # Default to 7 if type not recognized

def merge_related_chunks(chunks, similarity_threshold=0.8):
    """Merge chunks that are related and close in the document."""
    if not chunks:
        return chunks
    
    merged_chunks = []
    current_chunk = chunks[0]
    
    for next_chunk in chunks[1:]:
        # Check if chunks are from the same company and year
        same_document = (current_chunk['company'] == next_chunk['company'] and 
                        current_chunk['year'] == next_chunk['year'])
        
        # Check if chunks are sequential
        sequential = (next_chunk['chunk_index'] - current_chunk['chunk_index'] == 1)
        
        # Check similarity between chunks
        similarity = calculate_chunk_similarity(current_chunk['chunk_text'], 
                                             next_chunk['chunk_text'])
        
        if same_document and sequential and similarity > similarity_threshold:
            # Merge chunks
            current_chunk['chunk_text'] += "\n" + next_chunk['chunk_text']
            current_chunk['similarity'] = max(current_chunk['similarity'], 
                                            next_chunk['similarity'])
        else:
            merged_chunks.append(current_chunk)
            current_chunk = next_chunk
    
    merged_chunks.append(current_chunk)
    return merged_chunks

def calculate_chunk_similarity(text1, text2):
    """Calculate similarity between two chunks using their embeddings."""
    try:
        emb1 = get_embedding(text1)
        emb2 = get_embedding(text2)
        
        # Convert to numpy arrays
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    except Exception as e:
        logger.error(f"Error calculating chunk similarity: {e}")
        return 0.0

def query_faiss(faiss_index, metadata_store, query, max_results=CONFIG["MAX_RESULTS"], 
                similarity_threshold=CONFIG["SIMILARITY_THRESHOLD"],
                selected_companies=None):
    """Query the FAISS index with a text query and return relevant results."""
    # Analyze query type and get appropriate chunk count
    query_type = analyze_query_type(query)
    chunk_count = get_chunk_count(query_type)
    
    # Generate embedding for the query
    query_embedding = get_embedding(query)
    if not query_embedding:
        return []
    
    # Convert to numpy array and reshape for FAISS
    query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    
    # Search the FAISS index with more results for filtering
    distances, indices = faiss_index.search(query_vector, chunk_count * 2)
    
    # Process results
    results = []
    seen_texts = set()  # For deduplication
    
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < 0 or idx >= len(metadata_store):
            continue
        
        # Convert L2 distance to similarity score
        max_distance = np.sqrt(2)
        similarity = 1 - (distance / max_distance)
        
        if similarity < similarity_threshold:
            continue
        
        # Get metadata for this result
        metadata = metadata_store[idx]
        
        # Apply company filter if specified
        if selected_companies and metadata.get('company') not in selected_companies:
            continue
        
        # Deduplication check
        chunk_text = metadata["chunk_text"]
        if chunk_text in seen_texts:
            continue
        seen_texts.add(chunk_text)
        
        # Add to results
        results.append({
            "chunk_text": chunk_text,
            "company": metadata["company"],
            "year": metadata.get("year", "Unknown"),
            "chunk_index": metadata.get("chunk_index", 0),
            "similarity": float(similarity)
        })
    
    # Merge related chunks
    results = merge_related_chunks(results)
    
    # Limit to requested number of results
    return results[:chunk_count]

def answer_question_with_llm(query, results):
    """Use an LLM to answer a question based on the retrieved results."""
    # Group results by company
    company_results = {}
    for r in results:
        company = r['company']
        if company not in company_results:
            company_results[company] = []
        company_results[company].append(r)
    
    # Prepare context with company grouping
    context_parts = []
    for company, company_data in company_results.items():
        company_context = "\n\n".join([
            f"[Source: {r['company']} {r['year']}, Similarity: {r['similarity']:.3f}]\n{r['chunk_text']}"
            for r in company_data
        ])
        context_parts.append(f"=== {company} ===\n{company_context}")
    
    context = "\n\n".join(context_parts)
    
    # Create prompt for the LLM
    prompt = f"""
    You are an AI assistant that provides clear, contextual answers about company annual reports.
    
    QUESTION: {query}
    
    CONTEXT:
    {context}
    
    Please provide a comprehensive analysis following these guidelines:

    1. COMPANY OVERVIEW:
       - Clearly identify each company discussed
       - Provide relevant time periods and report dates
    
    2. FINANCIAL METRICS:
       - Group related financial metrics together
       - Explain what each metric means and why it's important
       - Highlight year-over-year changes with percentages
       - Put numbers in context (e.g., "revenue grew 20%, indicating strong market performance")
    
    3. COMPARATIVE ANALYSIS (if multiple companies):
       - Compare similar metrics across companies
       - Highlight significant differences or similarities
       - Consider industry context if available
    
    4. TRENDS AND INSIGHTS:
       - Identify key trends in the data
       - Explain the significance of major changes
       - Provide business context for the numbers
    
    5. VISUALIZATION SUGGESTIONS:
       Add a note about potential visualizations that would be helpful, such as:
       - Time series for trend analysis
       - Bar charts for company comparisons
       - Pie charts for revenue breakdown
       - Growth rate comparisons
    
    6. DATA COMPLETENESS:
       - Note any missing important metrics
       - Acknowledge if certain comparisons aren't possible
       - Highlight any time period gaps
    
    Format the response with clear sections and bullet points for readability.
    Focus on telling the financial story behind the numbers.
    If different companies are present, organize the information by company first, then by metric type.
    Do not make up information or use knowledge outside the provided context.
    
    ANSWER:
    """
    
    try:
        response = openai.chat.completions.create(
            model=CONFIG["LLM_MODEL"],
            messages=[
                {"role": "system", "content": """You are a financial analyst assistant that excels at:
                    - Breaking down complex financial metrics
                    - Providing industry context for numbers
                    - Explaining business implications
                    - Suggesting relevant visualizations
                    - Identifying trends and patterns
                    Always aim to make financial data understandable and actionable."""},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,  # Lower temperature for more focused responses
            max_tokens=1500   # Increased token limit for comprehensive analysis
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error getting LLM response: {e}")
        return "I encountered an error while trying to answer your question."

def main():
    st.set_page_config(
        page_title="RAG Document Search",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 RAG Document Search")
    
    # Load FAISS index and metadata
    if check_files_exist():
        faiss_index, metadata_store = load_faiss_index_and_metadata()
        if faiss_index is not None and metadata_store is not None:
            # Get available companies
            companies = get_metadata_summary(metadata_store)
            
            # Display metadata summary in an expander
            with st.expander("📚 Document Summary", expanded=True):
                st.text("Available Documents:")
                for company, count in companies.items():
                    st.text(f"• {company}: {count} chunks")
            
            # Add company filter in sidebar
            st.sidebar.subheader("Filter by Company")
            selected_companies = []
            for company in sorted(companies.keys()):
                if st.sidebar.checkbox(f"{company} ({companies[company]} chunks)", value=True):
                    selected_companies.append(company)
            
            if not selected_companies:
                st.sidebar.warning("Please select at least one company")
                st.stop()
    
    st.write("""
    This app follows the Query-Retrieve-Generate pattern:
    1. **Query**: Enter your search query
    2. **Retrieve**: Find relevant document chunks
    3. **Generate**: Get AI-generated answers based on retrieved content
    """)
    
    # Initialize session state for clear functionality
    if 'search_results' not in st.session_state:
        st.session_state['search_results'] = None
    if 'last_query' not in st.session_state:
        st.session_state['last_query'] = None
    
    # Check if files exist
    files_exist = check_files_exist()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # API Key configuration
    st.sidebar.subheader("OpenAI API Key")
    api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password", 
                                   value=os.getenv("OPENAI_API_KEY", ""))
    
    if api_key:
        # Save API key to .env file
        if save_api_key_to_env(api_key):
            # Set OpenAI API key
            openai.api_key = api_key
            st.sidebar.success("API key saved.")
        else:
            st.sidebar.error("Failed to save API key.")
    
    # Document processing
    st.sidebar.subheader("Document Processing")
    if not files_exist:
        st.sidebar.warning("FAISS index and metadata files not found.")
        if st.sidebar.button("Process Documents"):
            if run_document_processing():
                st.rerun()
    else:
        st.sidebar.success("FAISS index and metadata files found.")
    
    # Search configuration
    st.sidebar.subheader("Search Configuration")
    max_results = st.sidebar.slider("Maximum Results", 1, 10, CONFIG["MAX_RESULTS"])
    similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, CONFIG["SIMILARITY_THRESHOLD"], 0.1)
    
    # Create three columns for the RAG workflow
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.header("1️⃣ Query")
        st.write("Enter your search query to find relevant information.")
        
        # Clear button at the top of Query section
        if st.session_state['search_results'] is not None:
            if st.button("🗑️ Clear Previous Results", type="secondary"):
                st.session_state['search_results'] = None
                st.session_state['last_query'] = None
                st.rerun()
        
        # Sample queries
        st.subheader("Sample Queries")
        sample_query = st.selectbox("Select a sample query:", SAMPLE_QUERIES)
        
        # Query interface
        query = st.text_input("Enter your query:", value=sample_query, 
                            placeholder="e.g., What were the key financial metrics in 2023?")
        
        if query:
            with st.spinner("Searching..."):
                query_type = analyze_query_type(query)
                chunk_count = get_chunk_count(query_type)
                st.info(f"Query type: {query_type.title()} (using {chunk_count} chunks)")
                
                results = query_faiss(faiss_index, metadata_store, query, max_results, 
                                    similarity_threshold, selected_companies)
                
                if not results:
                    st.warning("No results found matching your query.")
                else:
                    st.success(f"Found {len(results)} relevant results!")
                    # Store results in session state
                    st.session_state['search_results'] = results
                    st.session_state['last_query'] = query
    
    with col2:
        st.header("2️⃣ Retrieve")
        st.write("View the retrieved document chunks that match your query.")
        
        if st.session_state['search_results'] is None:
            st.info("👈 Enter a query in the Query section to see retrieved results.")
        else:
            st.success(f"Retrieved {len(st.session_state['search_results'])} relevant chunks")
            
            # Display results with metadata
            for i, result in enumerate(st.session_state['search_results'], 1):
                with st.expander(f"Chunk {i} - {result['company']} ({result['year']}) - Similarity: {result['similarity']:.3f}"):
                    # Display metadata
                    st.write("**Metadata:**")
                    st.write(f"- Company: {result['company']}")
                    st.write(f"- Year: {result['year']}")
                    st.write(f"- Chunk Index: {result['chunk_index']}")
                    st.write(f"- Similarity Score: {result['similarity']:.3f}")
                    if 'genmab' in result:
                        st.write(f"- Genmab: {result['genmab']}")
                    st.write("---")
                    # Display content
                    st.write("**Content:**")
                    st.write(result['chunk_text'])
    
    with col3:
        st.header("3️⃣ Generate")
        st.write("Ask questions about the retrieved content to get AI-generated answers.")
        
        if st.session_state['search_results'] is None:
            st.info("👈 Enter a query and retrieve results first.")
        else:
            st.success(f"Ready to answer questions about: '{st.session_state['last_query']}'")
            
            # Clear button in Generate section
            if st.button("🗑️ Clear All Results", type="secondary"):
                st.session_state['search_results'] = None
                st.session_state['last_query'] = None
                st.rerun()
            
            # Generate answer for the original query
            if st.button("🤖 Generate Answer for Original Query", type="primary"):
                with st.spinner("Generating answer..."):
                    answer = answer_question_with_llm(st.session_state['last_query'], st.session_state['search_results'])
                    st.write("### Answer")
                    st.write(answer)
                    
                    # Show sources used
                    with st.expander("View Sources Used"):
                        for result in st.session_state['search_results']:
                            st.write(f"**{result['company']} ({result['year']})** - Similarity: {result['similarity']:.3f}")
                            st.write(result['chunk_text'])
                            st.write("---")
            
            # Divider for custom questions
            st.divider()
            st.write("Or ask a follow-up question:")
            
            # Custom question interface
            custom_question = st.text_input("Ask a different question:", 
                                          placeholder="e.g., What were the main challenges mentioned?")
            
            if custom_question:
                with st.spinner("Generating answer..."):
                    answer = answer_question_with_llm(custom_question, st.session_state['search_results'])
                    st.write("### Answer")
                    st.write(answer)
                    
                    # Show sources used
                    with st.expander("View Sources Used"):
                        for result in st.session_state['search_results']:
                            st.write(f"**{result['company']} ({result['year']})** - Similarity: {result['similarity']:.3f}")
                            st.write(result['chunk_text'])
                            st.write("---")

if __name__ == "__main__":
    main() 