import os
import sys
import json
import logging
import argparse
import openai
import faiss
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# ==============================================================================
# Logging Configuration
# ==============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("faiss_query.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Environment and Configuration
# ==============================================================================

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key from environment
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    logger.error("Warning: OPENAI_API_KEY not found in .env file")
    sys.exit(1)

# Configuration constants
CONFIG = {
    "EMBEDDING_DIM": 1536,
    "EMBEDDING_MODEL": "text-embedding-ada-002",
    "LLM_MODEL": "gpt-4.1-nano",
    "INDEX_PATH": "faiss.index",
    "METADATA_PATH": "metadata_store.json",
    "SIMILARITY_THRESHOLD": 0.7,
    "MAX_RESULTS": 5
}

# ==============================================================================
# Command-Line Arguments
# ==============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Query the FAISS-based vector store with metadata.')
    parser.add_argument('--index-path', type=str, default=CONFIG["INDEX_PATH"], help='Path to the FAISS index file')
    parser.add_argument('--metadata-path', type=str, default=CONFIG["METADATA_PATH"], help='Path to the metadata JSON file')
    parser.add_argument('--embedding-model', type=str, default=CONFIG["EMBEDDING_MODEL"], help='OpenAI embedding model to use')
    parser.add_argument('--llm-model', type=str, default=CONFIG["LLM_MODEL"], help='OpenAI LLM model to use')
    parser.add_argument('--max-results', type=int, default=CONFIG["MAX_RESULTS"], help='Maximum number of results to return')
    parser.add_argument('--similarity-threshold', type=float, default=CONFIG["SIMILARITY_THRESHOLD"], help='Minimum similarity threshold')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    return parser.parse_args()

# ==============================================================================
# FAISS and Metadata Loading
# ==============================================================================

def load_faiss_index_and_metadata(index_path, metadata_path):
    """
    Load the FAISS index and metadata from disk.
    
    Args:
        index_path (str): Path to the FAISS index file.
        metadata_path (str): Path to the metadata JSON file.
        
    Returns:
        tuple: (faiss_index, metadata_store)
    """
    try:
        # Load FAISS index
        logger.info(f"Loading FAISS index from '{index_path}'...")
        faiss_index = faiss.read_index(index_path)
        
        # Load metadata
        logger.info(f"Loading metadata from '{metadata_path}'...")
        with open(metadata_path, 'r') as f:
            metadata_store = json.load(f)
        
        logger.info(f"Successfully loaded FAISS index with {faiss_index.ntotal} vectors and {len(metadata_store)} metadata entries.")
        return faiss_index, metadata_store
    except Exception as e:
        logger.error(f"Error loading FAISS index and metadata: {e}")
        sys.exit(1)

# ==============================================================================
# Embedding and Query Functions
# ==============================================================================

def get_embedding(text, model=CONFIG["EMBEDDING_MODEL"]):
    """
    Generate an embedding for a single text via OpenAI API.
    
    Args:
        text (str): The text to embed.
        model (str): The OpenAI embedding model to use.
        
    Returns:
        list: The embedding vector.
    """
    try:
        response = openai.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

def query_faiss(faiss_index, metadata_store, query, max_results=CONFIG["MAX_RESULTS"], similarity_threshold=CONFIG["SIMILARITY_THRESHOLD"]):
    """
    Query the FAISS index with a text query and return relevant results.
    
    Args:
        faiss_index (faiss.Index): The loaded FAISS index.
        metadata_store (list): The loaded metadata store.
        query (str): The query text.
        max_results (int): Maximum number of results to return.
        similarity_threshold (float): Minimum similarity threshold.
        
    Returns:
        list: List of dictionaries with metadata and similarity scores.
    """
    # Generate embedding for the query
    query_embedding = get_embedding(query)
    if not query_embedding:
        logger.error("Failed to generate embedding for query.")
        return []
    
    # Convert to numpy array and reshape for FAISS
    query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    
    # Search the FAISS index
    distances, indices = faiss_index.search(query_vector, max_results)
    
    # Process results
    results = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < 0 or idx >= len(metadata_store):
            continue
        
        # Convert L2 distance to similarity score (1 - normalized distance)
        # FAISS uses L2 distance, so smaller is better
        max_distance = np.sqrt(2)  # Maximum possible L2 distance for normalized vectors
        similarity = 1 - (distance / max_distance)
        
        if similarity < similarity_threshold:
            continue
        
        # Get metadata for this result
        metadata = metadata_store[idx]
        
        # Add to results
        results.append({
            "chunk_text": metadata["chunk_text"],
            "company": metadata["company"],
            "year": metadata["year"],
            "chunk_index": metadata["chunk_index"],
            "similarity": float(similarity)
        })
    
    return results

def answer_question_with_llm(query, results, model=CONFIG["LLM_MODEL"]):
    """
    Use an LLM to answer a question based on the retrieved results.
    
    Args:
        query (str): The user's question.
        results (list): List of relevant chunks from the FAISS index.
        model (str): The OpenAI LLM model to use.
        
    Returns:
        str: The LLM's answer to the question.
    """
    # Prepare context from results
    context = "\n\n".join([
        f"[Source: {r['company']} {r['year']}, Chunk {r['chunk_index']}]\n{r['chunk_text']}"
        for r in results
    ])
    
    # Create prompt for the LLM
    prompt = f"""
    You are an AI assistant that answers questions based on information from annual reports.
    
    QUESTION: {query}
    
    CONTEXT:
    {context}
    
    Please answer the question based ONLY on the information provided in the context.
    If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question."
    Do not make up information or use knowledge outside the provided context.
    
    ANSWER:
    """
    
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on information from annual reports."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        return "I encountered an error while trying to answer your question."

# ==============================================================================
# Interactive Query Interface
# ==============================================================================

def display_results(results):
    """
    Display search results in a formatted way.
    
    Args:
        results (list): List of dictionaries with metadata and similarity scores.
    """
    if not results:
        print("No results found.")
        return
    
    print("\n" + "="*80)
    print("SEARCH RESULTS")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i} (Similarity: {result['similarity']:.3f})")
        print(f"Company: {result['company']} | Year: {result['year']} | Chunk: {result['chunk_index']}")
        print("-" * 80)
        text = result['chunk_text']
        if len(text) > 500:
            print(text[:500] + "...")
        else:
            print(text)
        print("="*80)

def interactive_query_loop(faiss_index, metadata_store, args):
    """
    Run an interactive loop for querying the FAISS index.
    
    Args:
        faiss_index (faiss.Index): The loaded FAISS index.
        metadata_store (list): The loaded metadata store.
        args: Command-line arguments.
    """
    print("\n" + "="*80)
    print("FAISS-BASED QUERY INTERFACE")
    print("="*80)
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'help' for available commands.")
    print("="*80)
    
    while True:
        query = input("\nEnter your question: ").strip()
        
        if query.lower() in ['exit', 'quit']:
            print("Exiting query interface. Goodbye!")
            break
        
        if query.lower() == 'help':
            print("\nAvailable commands:")
            print("  help - Show this help message")
            print("  exit, quit - Exit the program")
            print("\nYou can also ask questions about the annual reports, such as:")
            print("  - What were Pfizer's R&D expenses in 2022?")
            print("  - What was Moderna's revenue in 2023?")
            print("  - What were the key products for AstraZeneca in 2024?")
            continue
        
        if not query:
            continue
        
        print("\nSearching for relevant information...")
        results = query_faiss(
            faiss_index, 
            metadata_store, 
            query, 
            max_results=args.max_results,
            similarity_threshold=args.similarity_threshold
        )
        
        if not results:
            print("No relevant information found for your query.")
            continue
        
        # Display the raw results
        display_results(results)
        
        # Ask if the user wants an AI-generated answer
        answer_question = input("\nWould you like an AI-generated answer based on these results? (y/n): ").lower().strip()
        if answer_question in ['y', 'yes']:
            print("\nGenerating answer...")
            answer = answer_question_with_llm(query, results, model=args.llm_model)
            print("\n" + "="*80)
            print("AI ANSWER")
            print("="*80)
            print(answer)
            print("="*80)

# ==============================================================================
# Main Function
# ==============================================================================

def main():
    """
    Main function to load the FAISS index and metadata, then run the interactive query interface.
    """
    args = parse_arguments()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Load FAISS index and metadata
    faiss_index, metadata_store = load_faiss_index_and_metadata(args.index_path, args.metadata_path)
    
    # Run interactive query loop
    interactive_query_loop(faiss_index, metadata_store, args)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user. Exiting gracefully.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1) 