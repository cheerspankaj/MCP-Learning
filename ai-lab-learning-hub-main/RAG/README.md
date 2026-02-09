# RAG (Retrieval-Augmented Generation) Implementations

This directory contains different implementations of RAG (Retrieval-Augmented Generation) systems, each showcasing different approaches and technologies. These implementations serve as practical examples and learning resources for building RAG applications.

> 📚 **Learning Resources**: For a comprehensive understanding of RAG concepts, terminology, and best practices, see [RAG Concepts Guide](RAG_CONCEPTS.md).

## RAG Fundamentals

All RAG systems operate through two fundamental loops:

1. **Setup Loop** (one-time per corpus):
   - Ingest documents
   - Chunk them into segments
   - Generate embeddings
   - Store in a vector database

2. **Query Loop** (per user request):
   - Process the user's question
   - Retrieve relevant chunks
   - Generate a response using the context

Each implementation in this directory demonstrates these loops using different technologies and approaches.

## Available Implementations

### 1. OpenAI Managed RAG (`openai-managed-rag/`)
A streamlined implementation that leverages OpenAI's managed services for RAG:
- Uses OpenAI's file search and Assistants API
- Managed vector store and embeddings
- Simple setup with minimal infrastructure requirements
- Modern Streamlit interface for document querying
- Ideal for quick prototypes and small to medium-scale applications

### 2. DIY RAG with FAISS (`DIYRAG-FAISS/`)
A custom implementation using Facebook AI Similarity Search (FAISS):
- Local vector store using FAISS
- Custom document processing and chunking
- OpenAI embeddings integration
- Configurable similarity search parameters
- Complete control over the RAG pipeline
- Includes both CLI and Streamlit interfaces

### 3. LangChain RAG (`DIYRAG-langchain/`)
An implementation using the LangChain framework:
- Leverages LangChain's document loaders and text splitters
- Built-in vector store integrations
- Flexible chain composition
- Easy integration with different LLM providers
- Standardized RAG pipeline components

### 4. PostgreSQL RAG (`DIYRAG-postgres/`)  Coming soon
A database-centric implementation using PostgreSQL:
- Vector similarity search using pgvector
- Persistent storage of embeddings and metadata
- SQL-based querying capabilities
- Scalable for large document collections
- Integration with traditional database workflows

## Getting Started

Each implementation has its own README with specific setup instructions. Common prerequisites include:

1. Python 3.8+ environment
2. OpenAI API key
3. Required dependencies (specified in `requirements.txt`)
4. Basic understanding of RAG concepts

## Common Setup Steps

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```
4. Set up your OpenAI API key:
   ```bash
   echo "OPENAI_API_KEY=your-key-here" > .env
   ```

## Choosing an Implementation

- **Start with OpenAI Managed RAG** if you want:
  - Quick setup and deployment
  - Minimal infrastructure management
  - OpenAI's latest features
  
- **Use DIY FAISS** if you need:
  - Full control over the RAG pipeline
  - Local vector storage
  - Custom similarity search parameters
  
- **Choose LangChain** for:
  - Framework-based development
  - Multiple LLM provider support
  - Extensive RAG components
  
- **Consider PostgreSQL** when:
  - Integrating with existing databases
  - Need persistent vector storage
  - Working with large-scale applications

## Contributing

Feel free to contribute by:
- Adding new implementations
- Improving existing code
- Fixing bugs
- Enhancing documentation

## License

This project is licensed under the terms specified in the root directory's license file.

## Authors

- Farhat Siddiqui (FASI)
- Contributors from the AI Labs team 