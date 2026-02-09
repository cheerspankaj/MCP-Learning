# Building a RAG System: A Hands-on Tutorial

This tutorial walks you through building a prototype RAG system using FAISS for efficient similarity search and OpenAI for generating text. You'll learn how to implement two fundamental loops that power RAG systems: 

1. **Load-Chunk-Embed-Store Loop**: Processing documents into searchable embeddings
2. **Query-Retrieve-Generate Loop**: Using the processed data to answer questions

> 📚 **RAG Concepts**: For a deeper understanding of RAG concepts, terminology, and best practices, see the [RAG Concepts Guide](../RAG_CONCEPTS.md).

## 📚 Understanding RAG

RAG (Retrieval-Augmented Generation) combines the power of:
- Large Language Models (LLMs) for understanding and generating text
- Vector databases for efficient similarity search
- Your own document collection as a knowledge base

### Why RAG?
- Ground LLM responses in your specific data
- Reduce hallucination by providing context
- Keep information up-to-date without retraining
- Maintain control over knowledge sources

## 🛠️ Prerequisites

### System Requirements
- Python 3.8+
- 8GB+ RAM recommended
- 1GB+ free disk space
- CUDA-capable GPU (optional, for FAISS GPU support)

### Required Packages
```bash
uv sync
```
Key dependencies:
- `faiss-cpu` (or `faiss-gpu`): Vector similarity search
- `openai`: API access for embeddings and LLM
- `streamlit`: Web interface
- `PyPDF2`: PDF processing
- `python-dotenv`: Environment management

### API Keys
- OpenAI API key with access to:
  - `text-embedding-ada-002` model
  - GPT-4 or similar for text generation
- Store in `.env` file:
  ```bash
  OPENAI_API_KEY=your-key-here
  ```

### Document Requirements
- **Supported Formats**: PDF (text-based, not scanned)
- **File Naming**: `company_name_year.pdf` (e.g., `moderna_2023.pdf`)
- **File Size**: < 50MB per document recommended
- **Text Quality**: Machine-readable text (not images of text)
- **Location**: `data/annual_reports/` directory

### Supplemental Resources
To run the FAISS RAG examples, you'll need to download additional resources:
1. For running the first loop: Download the raw data files
2. For running only the second loop: Download the metadata and FAISS index (only needed if you want to skip loop 1, as running loop 1 will generate these files)

Download the required files from [here](https://microsoftgenmab.sharepoint.com/:f:/r/sites/aiproductschatgptqa/Shared%20Documents/Training/AI%20Learning%20Hub%20Unit/RAG%20supplemental%20resources?csf=1&web=1&e=0NqekN) and unzip the selected file inside the DIYRAY-FAISS folder.

## 🔄 Loop 1: Load-Chunk-Embed-Store

This first loop processes your documents into searchable chunks. Let's break down each step:

### 1. Load (`load_chunk_embed_store.py`)
```python
# Load PDF documents
documents = load_pdf_documents("data/annual_reports")
```
- Reads PDF files from your data directory
- Extracts text while preserving structure
- Handles multiple document formats

### 2. Chunk
```python
chunks = chunk_documents(documents, 
                        chunk_size=500,
                        overlap=50)
```
- Splits documents into manageable pieces
- Maintains context with overlap
- Preserves document metadata

### 3. Embed
```python
embeddings = generate_embeddings(chunks,
                               model="text-embedding-ada-002")
```
- Converts text chunks to vector embeddings
- Uses OpenAI's embedding model
- Creates numerical representations of text

### 4. Store
```python
# Store embeddings and metadata
faiss.write_index(index, "faiss.index")
json.dump(metadata, "metadata_store.json")
```
- Stores embeddings in FAISS index (vector store)
- Saves metadata separately for context
- Persists data for future queries

## 🔄 Loop 2: Query-Retrieve-Generate

This second loop handles user interactions. Let's explore each step:

### 1. Query (`rag_app.py`)
```python
query_embedding = get_embedding(user_query)
```
- Processes user's natural language query
- Converts query to embedding
- Handles query analysis and type detection

### 2. Retrieve
```python
results = query_faiss(faiss_index,
                     query_embedding,
                     top_k=5)
```
- Searches FAISS index for similar chunks
- Filters by metadata (e.g., company)
- Merges related chunks

### 3. Generate
```python
answer = generate_response(query,
                         results,
                         model="gpt-4")
```
- Provides context to LLM
- Structures comprehensive responses
- Includes analysis and visualizations

## 🚀 Building Your Own RAG System

### Prerequisites
- Python 3.8+
- OpenAI API key
- PDF documents to process

### Step-by-Step Tutorial

1. **Setup Your Environment**
   ```bash
   git clone <repository-url>
   cd RAG/DIYRAG-FAISS
   pip install -r requirements.txt
   ```

2. **Prepare Your Documents**
   - Create `data/annual_reports` directory
   - Add PDF files
   - Structure: `data/annual_reports/company_name_year.pdf`

3. **Run Load-Chunk-Embed Loop**
   ```bash
   python load_chunk_embed_store.py
   ```
   This creates:
   - FAISS index (`faiss.index`)
   - Metadata store (`metadata_store.json`)
   - Processing logs (`faiss_processing.log`)

4. **Launch Query-Retrieve-Generate Interface**
   ```bash
   streamlit run rag_app.py
   ```
   Features:
   - Company filtering
   - Dynamic chunk selection
   - Smart result merging
   - Contextual analysis

## 💡 Advanced Features

### Smart Chunk Processing
- Dynamic chunk sizing based on content
- Overlap for context preservation
- Deduplication and merging

### Intelligent Retrieval
- Query type detection
- Adaptive result count
- Company-specific filtering

### Enhanced Response Generation
- Financial metric analysis
- Trend identification
- Visualization suggestions
- Data completeness checks

## 🔍 Example Queries

Try these queries to explore the system:
```
"What were the key financial metrics in 2023?"
"Compare R&D investments across companies"
"Explain the revenue growth trends"
```

## 📊 Understanding the Results

The system provides:
1. **Company Overview**
   - Clear company identification
   - Relevant time periods

2. **Financial Analysis**
   - Grouped metrics
   - Context and implications
   - Year-over-year changes

3. **Visualization Suggestions**
   - Relevant chart types
   - Data presentation ideas

4. **Data Completeness**
   - Missing metrics noted
   - Time period gaps identified

## 🛠️ Customization

### Adjusting Chunk Size
```python
# In load_chunk_embed_store.py
CONFIG = {
    "CHUNK_SIZE": 500,
    "CHUNK_OVERLAP": 50
}
```

### Modifying Search Parameters
```python
# In rag_app.py
CONFIG = {
    "MAX_RESULTS": 5,
    "SIMILARITY_THRESHOLD": 0.7
}
```

## 📝 Best Practices

1. **Document Processing**
   - Clean and consistent formatting
   - Appropriate chunk sizes
   - Meaningful metadata

2. **Query Design**
   - Clear and specific questions
   - Use company filters when needed
   - Consider query type

3. **Result Analysis**
   - Check data completeness
   - Verify time periods
   - Compare across sources

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional document formats
- More visualization types
- Enhanced analysis features

## 📚 Learn More

- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Streamlit Guides](https://docs.streamlit.io)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚙️ Configuration Guide

### Chunk Size and Overlap
```python
CONFIG = {
    "CHUNK_SIZE": 500,    # Characters per chunk
    "CHUNK_OVERLAP": 50   # Characters of overlap
}
```
- **Chunk Size**: 
  - Smaller (200-300): Better for precise queries, more chunks
  - Larger (500-1000): Better for context, fewer chunks
  - Default (500): Good balance for most documents

- **Overlap**: 
  - Smaller (25-50): Efficient storage, might miss cross-chunk context
  - Larger (50-100): Better context, more storage needed
  - Default (50): Good balance for most cases

### Search Parameters
```python
CONFIG = {
    "MAX_RESULTS": 5,              # Top-k chunks to retrieve
    "SIMILARITY_THRESHOLD": 0.7,    # Minimum similarity score
    "EMBEDDING_MODEL": "text-embedding-ada-002",
    "LLM_MODEL": "gpt-4"
}
```
- **MAX_RESULTS**: 
  - Lower (3-5): More focused, might miss context
  - Higher (7-10): More context, might include noise
  - Adjust based on query complexity

- **SIMILARITY_THRESHOLD**:
  - Lower (0.5-0.6): More results, less relevant
  - Higher (0.8-0.9): Fewer results, more relevant
  - Default (0.7): Good balance

## 🔍 Troubleshooting

### Common Issues

1. **FAISS Index Errors**
   ```
   Error: Could not load FAISS index
   ```
   - Check file permissions
   - Verify index file exists
   - Ensure enough memory

2. **Embedding Generation Failures**
   ```
   Error: OpenAI API error
   ```
   - Verify API key
   - Check rate limits
   - Confirm model access

3. **PDF Processing Issues**
   ```
   Error: Could not extract text from PDF
   ```
   - Ensure PDF is text-based
   - Check file corruption
   - Verify file permissions

### Performance Optimization

1. **Memory Usage**
   - Use `faiss-gpu` for large indices
   - Adjust chunk size for memory constraints
   - Monitor RAM usage

2. **Search Speed**
   - Index optimization
   - Batch processing
   - GPU acceleration

3. **Result Quality**
   - Fine-tune similarity threshold
   - Adjust chunk size and overlap
   - Refine query processing

## 📊 Monitoring and Logging

The system maintains several log files:
- `faiss_processing.log`: Document processing events
- `rag_streamlit.log`: Query and response tracking
- `error.log`: Error tracking and debugging

Monitor these for:
- Processing success/failure
- Query patterns
- Error trends
- Performance metrics 