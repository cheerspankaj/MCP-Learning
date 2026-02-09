import os
import re
import openai
import sys
import time
import argparse
import tiktoken
import logging
from pathlib import Path
from dotenv import load_dotenv
import PyPDF2
import faiss
import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm
import traceback

# ==============================================================================
# Logging Configuration
# ==============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("faiss_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Environment and FAISS Initialization
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
    "MAX_TOKENS_PER_CHUNK": 500,
    "OVERLAP_TOKENS": 100,
    "BATCH_SIZE": 100,
    "EMBEDDING_MODEL": "text-embedding-ada-002",
    "LLM_MODEL": "gpt-4.1-nano",
    "DATA_DIR": "./data/annual_reports",
    "INDEX_PATH": "faiss.index",
    "METADATA_PATH": "metadata_store.json",
    "SIMILARITY_THRESHOLD": 0.7
}

# FAISS index configuration
faiss_index = faiss.IndexFlatL2(CONFIG["EMBEDDING_DIM"])

# Global metadata store to keep track of chunk metadata
metadata_store = []
current_id = 0  # To assign unique IDs to each chunk

# ==============================================================================
# Command-Line Arguments
# ==============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process PDF files and store embeddings in a FAISS index with metadata.')
    parser.add_argument('--skip-discovery', action='store_true', help='Skip PDF file discovery')
    parser.add_argument('--skip-metadata', action='store_true', help='Skip metadata extraction')
    parser.add_argument('--skip-chunking', action='store_true', help='Skip text extraction and chunking')
    parser.add_argument('--skip-embedding', action='store_true', help='Skip embedding generation and storage')
    parser.add_argument('--non-interactive', action='store_true', help='Run without asking for confirmation')
    parser.add_argument('--data-dir', type=str, default=CONFIG["DATA_DIR"], help='Directory containing PDF files')
    parser.add_argument('--index-path', type=str, default=CONFIG["INDEX_PATH"], help='Path to save FAISS index')
    parser.add_argument('--metadata-path', type=str, default=CONFIG["METADATA_PATH"], help='Path to save metadata')
    parser.add_argument('--max-tokens', type=int, default=CONFIG["MAX_TOKENS_PER_CHUNK"], help='Maximum tokens per chunk')
    parser.add_argument('--overlap-tokens', type=int, default=CONFIG["OVERLAP_TOKENS"], help='Number of tokens to overlap between chunks')
    parser.add_argument('--batch-size', type=int, default=CONFIG["BATCH_SIZE"], help='Batch size for embedding generation')
    parser.add_argument('--embedding-model', type=str, default=CONFIG["EMBEDDING_MODEL"], help='OpenAI embedding model to use')
    parser.add_argument('--llm-model', type=str, default=CONFIG["LLM_MODEL"], help='OpenAI LLM model to use')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    return parser.parse_args()

# ==============================================================================
# Embedding and Chunking Functions
# ==============================================================================

def get_embedding(text, model=CONFIG["EMBEDDING_MODEL"]):
    """Generate an embedding for a single text via OpenAI API."""
    try:
        response = openai.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

def batch_get_embeddings(texts, model=CONFIG["EMBEDDING_MODEL"], batch_size=CONFIG["BATCH_SIZE"]):
    """
    Generate embeddings for multiple texts in batches via OpenAI API.
    
    Args:
        texts (list): List of text strings to get embeddings for.
        model (str): OpenAI embedding model to use.
        batch_size (int): Number of texts to process in each API call.
        
    Returns:
        list: List of embeddings in the same order as input texts.
    """
    try:
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        logger.info(f"Generating embeddings in {total_batches} batches of {batch_size}...")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            current_batch = (i // batch_size) + 1
            
            response = openai.embeddings.create(
                model=model,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
            if i + batch_size < len(texts):
                time.sleep(0.1)  # Rate limiting
        
        logger.info(f"All {len(texts)} embeddings generated successfully!")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        logger.error(traceback.format_exc())
        return None

def get_token_count(text, model=CONFIG["EMBEDDING_MODEL"]):
    """Get the number of tokens in a text string using tiktoken."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def chunk_text(text, max_tokens=CONFIG["MAX_TOKENS_PER_CHUNK"], overlap_tokens=CONFIG["OVERLAP_TOKENS"]):
    """
    Split text into overlapping chunks based on token count.
    
    Args:
        text (str): The text to split into chunks.
        max_tokens (int): Maximum tokens per chunk (default: 500).
        overlap_tokens (int): Number of tokens to overlap (default: 100).
    
    Returns:
        list[str]: List of text chunks with overlap between consecutive chunks.
    """
    encoding = tiktoken.encoding_for_model(CONFIG["EMBEDDING_MODEL"])
    
    # Normalize whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    # Split text into sentences
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = len(encoding.encode(sentence))
        
        # If the sentence is too long, further split it on punctuation or spaces
        if sentence_tokens > max_tokens:
            parts = re.split(r'(?<=[,;:])\s+', sentence)
            for part in parts:
                part_tokens = len(encoding.encode(part))
                if part_tokens > max_tokens:
                    words = part.split()
                    temp_chunk = []
                    temp_tokens = 0
                    for word in words:
                        word_tokens = len(encoding.encode(word))
                        if temp_tokens + word_tokens + 1 > max_tokens:
                            chunks.append(' '.join(temp_chunk))
                            temp_chunk = [word]
                            temp_tokens = word_tokens
                        else:
                            temp_chunk.append(word)
                            temp_tokens += word_tokens + 1
                    if temp_chunk:
                        chunks.append(' '.join(temp_chunk))
                else:
                    if current_tokens + part_tokens + 1 > max_tokens:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [part]
                        current_tokens = part_tokens
                    else:
                        current_chunk.append(part)
                        current_tokens += part_tokens + 1
        else:
            if current_tokens + sentence_tokens + 1 > max_tokens:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    # Add overlaps to maintain context
    if overlap_tokens > 0:
        overlapped_chunks = []
        for i in range(len(chunks)):
            if i == 0:
                overlapped_chunks.append(chunks[i])
            else:
                prev_chunk = chunks[i-1]
                prev_tokens = encoding.encode(prev_chunk)
                if len(prev_tokens) > overlap_tokens:
                    overlap_tokens_list = prev_tokens[-overlap_tokens:]
                    overlap_text = encoding.decode(overlap_tokens_list)
                else:
                    overlap_text = prev_chunk
                overlapped_chunks.append(overlap_text + " " + chunks[i])
        chunks = overlapped_chunks
    
    # Debug: Print token count per chunk
    for i, chunk in enumerate(chunks):
        token_count = len(encoding.encode(chunk))
        logger.debug(f"Chunk {i}: {token_count} tokens")
    
    return chunks

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file using PyPDF2."""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logger.error(f"Error reading {pdf_path}: {e}")
        logger.error(traceback.format_exc())
    return text

# ==============================================================================
# Metadata Extraction Functions
# ==============================================================================

def extract_metadata_with_llm(pdf_path, first_page_text):
    """
    Extract company name and year from PDF using an LLM.
    This function is used for direct extraction or as a fallback.
    """
    try:
        # Create a more specific prompt that emphasizes the importance of the filename
        # and key phrases like "year ending"
        prompt = f"""
        Extract the company name and year from this annual report PDF.
        
        FILENAME: {pdf_path.name}
        
        The filename often contains the correct year, especially if it follows the pattern:
        "company_name year uuid.pdf" (e.g., "moderna 2024 12d8720a-9f51-4695-b0e9-f2d45dff1c69.pdf")
        
        First page text:
        {first_page_text[:2000]}
        
        IMPORTANT CLUES FOR THE YEAR:
        1. Look for phrases like "year ending December 31, 2024" or "for the year ended December 31, 2024"
        2. The phrase "year ending" or "year ended" followed by a date is a strong indicator of the report year
        3. If the filename contains a year (like "moderna 2024"), that is likely the correct year
        4. Look for "Annual Report 2024" or similar phrases
        
        Return ONLY a JSON object with these fields:
        {{
            "company": "Company name (e.g., AstraZeneca, Moderna, Pfizer)",
            "year": "Year as a number (e.g., 2020, 2021)"
        }}
        """
        response = openai.chat.completions.create(
            model=CONFIG["LLM_MODEL"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant extracting metadata from financial documents. Pay special attention to the filename and phrases like 'year ending' which indicate the report year."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=150
        )
        result_text = response.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            try:
                metadata = json.loads(json_match.group(0))
                return metadata.get("company"), metadata.get("year")
            except json.JSONDecodeError:
                logger.error(f"Error parsing JSON from LLM response: {result_text}")
        return None, None
    except Exception as e:
        logger.error(f"Error using LLM for metadata extraction: {e}")
        logger.error(traceback.format_exc())
        return None, None

def validate_metadata_with_llm(pdf_path, first_page_text, extracted_company, extracted_year):
    """
    Validate extracted metadata using an LLM.
    
    Args:
        pdf_path (Path): Path to the PDF file.
        first_page_text (str): Text extracted from the first page.
        extracted_company (str): Company name extracted using regex.
        extracted_year (str): Year extracted using regex.
        
    Returns:
        tuple: (validated_company, validated_year)
    """
    try:
        prompt = f"""
        I've extracted metadata from this annual report PDF, but I need you to validate it.
        
        Filename: {pdf_path.name}
        
        Extracted company: {extracted_company}
        Extracted year: {extracted_year}
        
        First page text:
        {first_page_text[:2000]}
        
        Please validate the company name and year. If they're correct, return them as is.
        If they're incorrect, provide the correct values.
        
        Return ONLY a JSON object with these fields:
        {{
            "company": "Company name (e.g., AstraZeneca, Moderna, Pfizer)",
            "year": "Year as a number (e.g., 2020, 2021)"
        }}
        """
        response = openai.chat.completions.create(
            model=CONFIG["LLM_MODEL"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant validating metadata from financial documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=150
        )
        result_text = response.choices[0].message.content.strip()
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            try:
                metadata = json.loads(json_match.group(0))
                return metadata.get("company"), metadata.get("year")
            except json.JSONDecodeError:
                logger.error(f"Error parsing JSON from LLM response: {result_text}")
        return extracted_company, extracted_year
    except Exception as e:
        logger.error(f"Error using LLM for metadata validation: {e}")
        logger.error(traceback.format_exc())
        return extracted_company, extracted_year

def extract_metadata_from_first_page(pdf_path):
    """
    Extract company name and year from the first page of a PDF.
    Uses regex-based extraction and filename patterns, with LLM extraction for edge cases.
    """
    first_page_text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            if reader.pages:
                first_page_text = reader.pages[0].extract_text() or ""
                if len(reader.pages) > 1:
                    first_page_text += "\n" + (reader.pages[1].extract_text() or "")
    except Exception as e:
        logger.error(f"Error reading first page of {pdf_path}: {e}")
        logger.error(traceback.format_exc())
        return None, None

    filename = pdf_path.name.lower()
    
    # Special case for Pfizer files - they often have specific naming patterns
    if "pfizer" in filename:
        logger.info(f"Special handling for Pfizer file: {pdf_path.name}")
        
        # Look for year in the filename after "pfizer"
        pfizer_year_match = re.search(r'pfizer\s+(\d{4})', filename)
        if pfizer_year_match:
            year_found = pfizer_year_match.group(1)
            logger.info(f"Found year {year_found} after 'pfizer' in filename")
            return "Pfizer", year_found
        
        # If no year found after "pfizer", try to find any year in the filename
        year_matches = re.findall(r'20[1-3]\d', filename)
        if year_matches:
            year_found = year_matches[-1]  # Take the last year found
            logger.info(f"Found year {year_found} in Pfizer filename")
            return "Pfizer", year_found
        
        # If still no year found, use LLM extraction
        logger.info(f"Using LLM for Pfizer file: {pdf_path.name}")
        llm_company, llm_year = extract_metadata_with_llm(pdf_path, first_page_text)
        if llm_company and llm_year:
            logger.info(f"LLM extracted for Pfizer: Company = {llm_company}, Year = {llm_year}")
            return llm_company, llm_year
    
    # Check for unusual filename patterns that might benefit from LLM extraction
    unusual_filename_patterns = [
        r'\d{6}',  # Date-like patterns (e.g., 120225)
        r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}',  # UUID patterns
        r'\d{4}\s+\d{4}',  # Multiple year-like numbers
    ]
    
    has_unusual_pattern = any(re.search(pattern, filename) for pattern in unusual_filename_patterns)
    
    # For files with unusual patterns, use LLM extraction directly
    if has_unusual_pattern:
        logger.info(f"Using LLM for direct extraction due to unusual filename pattern: {pdf_path.name}")
        llm_company, llm_year = extract_metadata_with_llm(pdf_path, first_page_text)
        if llm_company and llm_year:
            logger.info(f"LLM extracted: Company = {llm_company}, Year = {llm_year}")
            return llm_company, llm_year
    
    # Otherwise, proceed with regex-based extraction
    company_from_filename = None

    company_identifiers = {
        "moderna": {
            "filename_patterns": ["moderna"],
            "text_patterns": ["moderna", "delaware 81-3467528"],
            "output_name": "Moderna"
        },
        "pfizer": {
            "filename_patterns": ["pfizer"],
            "text_patterns": ["pfizer", "delaware 13-5315170"],
            "output_name": "Pfizer"
        },
        "astrazeneca": {
            "filename_patterns": ["astrazeneca", "astra-zeneca"],
            "text_patterns": ["astrazeneca", "astra zeneca", "astra-zeneca"],
            "output_name": "AstraZeneca"
        },
        "genmab": {
            "filename_patterns": ["genmab", "gmab"],
            "text_patterns": ["genmab", "genmab a/s"],
            "output_name": "Genmab"
        },
        "regeneron": {
            "filename_patterns": ["regn", "regeneron"],
            "text_patterns": ["regeneron", "regeneron pharmaceuticals"],
            "output_name": "Regeneron"
        },
        "roche": {
            "filename_patterns": ["roche"],
            "text_patterns": [
                "roche", "f. hoffmann-la roche", "f hoffmann la roche",
                "hoffmann la roche", "roche group", "roche holding"
            ],
            "output_name": "Roche"
        }
    }
    
    # Try filename patterns first
    for company, patterns in company_identifiers.items():
        if any(pattern in filename for pattern in patterns["filename_patterns"]):
            company_from_filename = patterns["output_name"]
            break

    # Then try text patterns
    company_from_text = None
    text_lower = first_page_text.lower()
    for company, patterns in company_identifiers.items():
        if any(pattern in text_lower for pattern in patterns["text_patterns"]):
            company_from_text = patterns["output_name"]
            break

    company_name = company_from_filename or company_from_text

    # Extract year using filename or text patterns
    year_found = None
    
    # First, look for year in the filename with a more specific pattern
    # This will match patterns like "2024-annual-report.pdf" or "annual-report-2024.pdf"
    year_patterns = [
        r'(?:annual-report-)(\d{4})',  # annual-report-2024.pdf
        r'(\d{4})(?:-annual-report)',  # 2024-annual-report.pdf
        r'(?:report-)(\d{4})',         # report-2024.pdf
        r'(\d{4})(?:-report)',         # 2024-report.pdf
        r'(?:ar-)(\d{4})',             # ar-2024.pdf
        r'(\d{4})(?:-ar)',             # 2024-ar.pdf
    ]
    
    for pattern in year_patterns:
        match = re.search(pattern, filename)
        if match:
            year_found = match.group(1)
            logger.debug(f"Found year {year_found} in filename using pattern '{pattern}'")
            break
    
    # If no year found in filename with specific patterns, try a more general approach
    if not year_found:
        # Look for 4-digit years in the filename
        year_matches = re.findall(r'20[1-3]\d', filename)
        if year_matches:
            # If multiple years found, prefer the one that appears after "annual" or "report"
            for year in year_matches:
                if re.search(r'(?:annual|report).*' + year, filename):
                    year_found = year
                    logger.debug(f"Found year {year_found} in filename after 'annual' or 'report'")
                    break
            
            # If no preference found, use the last year in the filename
            if not year_found:
                year_found = year_matches[-1]
                logger.debug(f"Using last year {year_found} found in filename")
    
    # Special case for filenames like "pfizer 2021 9353920d-d7a2-4609-9685-f302fe931480.pdf"
    if not year_found and company_from_filename:
        # Look for a year after the company name
        company_year_match = re.search(rf"{company_from_filename.lower()}\s+(\d{{4}})", filename)
        if company_year_match:
            year_found = company_year_match.group(1)
            logger.debug(f"Found year {year_found} after company name in filename")
    
    # If still no year found, try to extract from the text
    if not year_found:
        year_patterns = [
            r'(?:fiscal\s+year\s+ended\s+(?:December|Jan|February|March|April|May|June|July|August|September|October|November)\s+\d{1,2},?\s+)(20[1-3]\d)',
            r'(?:Annual\s+Report\s+)(20[1-3]\d)',
            r'(?:December\s+31,\s+)(20[1-3]\d)',
            r'(?:31\s+December\s+)(20[1-3]\d)',
            r'(?:31\.12\.)(20[1-3]\d)',
            r'\b(20[1-3]\d)\b'
        ]
        for pattern in year_patterns:
            match = re.search(pattern, first_page_text)
            if match:
                year_found = match.group(1)
                logger.debug(f"Found year {year_found} in text using pattern '{pattern}'")
                break
    
    # If we have both company and year, validate with LLM
    if company_name and year_found:
        # Always validate if the filename contains date-like patterns (e.g., 120225)
        if "120225" in filename or re.search(r'\d{6}', filename):
            logger.info(f"Validating metadata for {pdf_path.name} with LLM (date-like pattern detected)")
            validated_company, validated_year = validate_metadata_with_llm(pdf_path, first_page_text, company_name, year_found)
            
            if validated_company != company_name:
                logger.info(f"LLM corrected company: {company_name} -> {validated_company}")
                company_name = validated_company
                
            if validated_year != year_found:
                logger.info(f"LLM corrected year: {year_found} -> {validated_year}")
                year_found = validated_year
    
    # If we have a company but no year, use LLM to extract the year
    elif company_name and not year_found:
        logger.info(f"Using LLM to extract year for {pdf_path.name}")
        _, llm_year = extract_metadata_with_llm(pdf_path, first_page_text)
        if llm_year:
            year_found = llm_year
            logger.info(f"LLM found year: {year_found}")
    
    # If we have a year but no company, use LLM to extract the company
    elif year_found and not company_name:
        logger.info(f"Using LLM to extract company for {pdf_path.name}")
        llm_company, _ = extract_metadata_with_llm(pdf_path, first_page_text)
        if llm_company:
            company_name = llm_company
            logger.info(f"LLM found company: {company_name}")

    # Debug printing
    logger.debug(f"Debug - Results for {pdf_path.name}:")
    logger.debug(f"Company from filename: {company_from_filename}")
    logger.debug(f"Company from text: {company_from_text}")
    logger.debug(f"Final company: {company_name}")
    logger.debug(f"Year: {year_found}\n")

    return company_name, year_found

# ==============================================================================
# FAISS Storage Functions
# ==============================================================================

def store_chunk_faiss(company, year, chunk_index, chunk_text, embedding):
    """
    Add the embedding to the FAISS index and store metadata.
    
    Args:
        company (str): Company name.
        year (str/int): Report year.
        chunk_index (int): Index of the chunk within the document.
        chunk_text (str): The text chunk.
        embedding (list[float]): Embedding vector from OpenAI API.
        
    Returns:
        bool: True on success, False on error.
    """
    global faiss_index, metadata_store, current_id
    try:
        # Convert the embedding list to a numpy array of type float32
        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        faiss_index.add(vec)  # The add method takes the array directly
        # Store metadata with a unique ID
        metadata_store.append({
            "id": current_id,
            "company": company,
            "year": year,
            "chunk_index": chunk_index,
            "chunk_text": chunk_text
        })
        current_id += 1
        return True
    except Exception as e:
        logger.error(f"Error storing chunk in FAISS: {e}")
        logger.error(traceback.format_exc())
        return False

# ==============================================================================
# Document Processing Pipeline
# ==============================================================================

def process_report(file_path, company, year):
    """
    Process a single PDF report: extract text, chunk it, generate embeddings, and store in FAISS.
    
    Args:
        file_path (Path): Path to the PDF file.
        company (str): Company name extracted from the PDF.
        year (str/int): Report year extracted from the PDF.
        
    Returns:
        bool: True if at least one chunk was successfully stored.
    """
    logger.info(f"Processing {file_path.name} for {company} ({year})...")
    
    text = extract_text_from_pdf(file_path)
    if not text:
        logger.warning(f"No text extracted from {file_path.name}. Skipping.")
        return False
    
    # Prepend metadata to text
    text_with_metadata = f"Company: {company}\nYear: {year}\n\n{text}"
    
    chunks = chunk_text(text_with_metadata)
    logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
    
    # Generate embeddings for all chunks
    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = batch_get_embeddings(chunks)
    if not embeddings:
        logger.error(f"Failed to generate embeddings for {file_path.name}")
        return False
    
    # Store each chunk in FAISS along with metadata
    success_count = 0
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        logger.debug(f"Storing chunk {i+1}/{len(chunks)} for {company} {year}...")
        if store_chunk_faiss(company, year, i, chunk, embedding):
            success_count += 1
    
    logger.info(f"Successfully stored {success_count}/{len(chunks)} chunks for {file_path.name}")
    return success_count > 0

def print_faiss_processing_report():
    """Print a report summarizing the chunks stored in the FAISS index and metadata."""
    logger.info("\n" + "="*80)
    logger.info("FAISS INDEX PROCESSING REPORT")
    logger.info("="*80)
    
    total_chunks = len(metadata_store)
    logger.info(f"Total chunks stored in FAISS index: {total_chunks}")
    
    # Aggregate counts by company and year
    stats = defaultdict(lambda: {"chunks": 0})
    for md in metadata_store:
        key = (md["company"], md["year"])
        stats[key]["chunks"] += 1
    
    logger.info("\nBreakdown by Company and Year:")
    logger.info("-" * 80)
    logger.info(f"{'Company':<15} {'Year':<6} {'Chunks':<10}")
    for (company, year), data in stats.items():
        logger.info(f"{company:<15} {year:<6} {data['chunks']:<10}")
    
    logger.info("="*80 + "\n")

# ==============================================================================
# Persistence: Saving FAISS Index and Metadata to Disk
# ==============================================================================

def save_faiss_index_and_metadata(index_path=CONFIG["INDEX_PATH"], metadata_path=CONFIG["METADATA_PATH"]):
    """
    Save the FAISS index and metadata store to disk.
    
    Args:
        index_path (str): File path to store the FAISS index.
        metadata_path (str): File path to store the metadata (as JSON).
    """
    try:
        # Save FAISS index
        faiss.write_index(faiss_index, index_path)
        # Save metadata as JSON
        with open(metadata_path, 'w') as f:
            json.dump(metadata_store, f, indent=4)
        logger.info(f"FAISS index has been saved to '{index_path}'")
        logger.info(f"Metadata has been saved to '{metadata_path}'")
    except Exception as e:
        logger.error(f"Error saving FAISS index and metadata: {e}")
        logger.error(traceback.format_exc())

# ==============================================================================
# Main Pipeline
# ==============================================================================

def confirm_action(prompt):
    """
    Ask for user confirmation with y/n.
    
    Args:
        prompt (str): The prompt to display to the user.
        
    Returns:
        bool: True if user confirms, False otherwise.
    """
    while True:
        response = input(f"{prompt} (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'.")

def print_metadata_summary(files_to_process):
    """
    Print a summary of all metadata found.
    
    Args:
        files_to_process (list): List of dictionaries containing file metadata.
    """
    logger.info("\n" + "="*80)
    logger.info("METADATA SUMMARY")
    logger.info("="*80)
    
    # Group by company
    company_groups = defaultdict(list)
    for entry in files_to_process:
        company_groups[entry["company"]].append(entry)
    
    # Print summary by company
    for company, entries in sorted(company_groups.items()):
        logger.info(f"\n{company}:")
        logger.info("-" * 40)
        # Convert all years to integers for consistent sorting
        for entry in sorted(entries, key=lambda x: int(x["year"]) if isinstance(x["year"], str) else x["year"]):
            logger.info(f"  - {entry['file'].name}: Year = {entry['year']}")
    
    logger.info("\n" + "="*80)

def main():
    """
    Main function to process PDFs into a FAISS-based vector store with metadata.
    Steps include file discovery, metadata extraction, text chunking, embedding generation, and FAISS storage.
    """
    args = parse_arguments()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Update configuration from command line arguments
    CONFIG["DATA_DIR"] = args.data_dir
    CONFIG["INDEX_PATH"] = args.index_path
    CONFIG["METADATA_PATH"] = args.metadata_path
    CONFIG["MAX_TOKENS_PER_CHUNK"] = args.max_tokens
    CONFIG["OVERLAP_TOKENS"] = args.overlap_tokens
    CONFIG["BATCH_SIZE"] = args.batch_size
    CONFIG["EMBEDDING_MODEL"] = args.embedding_model
    CONFIG["LLM_MODEL"] = args.llm_model
    
    logger.info("\n" + "="*80)
    logger.info("FAISS-BASED DOCUMENT PROCESSING PIPELINE")
    logger.info("="*80)
    
    # STEP 1: PDF File Discovery
    if args.skip_discovery:
        logger.info("\nSkipping PDF file discovery is not supported without a saved file list. Exiting.")
        sys.exit(1)
    else:
        logger.info("\nSTEP 1: PDF File Discovery")
        logger.info(f"Scanning for PDF files in the '{CONFIG['DATA_DIR']}' directory...")
        data_dir = Path(CONFIG["DATA_DIR"])
        if not data_dir.exists():
            logger.error(f"Error: Data directory '{data_dir}' does not exist. Please create it and add your PDF files.")
            sys.exit(1)
        pdf_files = list(data_dir.glob("*.pdf"))
        if not pdf_files:
            logger.error(f"No PDF files found in {data_dir}. Please add some PDF files and try again.")
            sys.exit(1)
        logger.info(f"Found {len(pdf_files)} PDF files:")
        for i, file in enumerate(pdf_files, 1):
            logger.info(f"  {i}. {file.name}")
        
        if not args.non_interactive:
            if not confirm_action("\nDo you want to proceed to metadata extraction?"):
                logger.info("Process cancelled by user. Exiting.")
                sys.exit(0)
    
    # STEP 2: Metadata Extraction
    if args.skip_metadata:
        logger.info("\nSkipping metadata extraction is not supported without saved metadata. Exiting.")
        sys.exit(1)
    else:
        logger.info("\nSTEP 2: Metadata Extraction")
        logger.info("Extracting metadata (company name and year) from each PDF...")
        files_to_process = []
        for file_path in tqdm(pdf_files, desc="Extracting metadata"):
            company, year = extract_metadata_from_first_page(file_path)
            if company and year:
                files_to_process.append({
                    "file": file_path,
                    "company": company,
                    "year": year
                })
                logger.info(f"✓ {file_path.name}: Company = {company}, Year = {year}")
            else:
                logger.warning(f"✗ {file_path.name} – Unable to extract valid metadata (company: {company}, year: {year}).")
        
        if not files_to_process:
            logger.error("No files with valid metadata found. Exiting.")
            sys.exit(1)
        
        logger.info(f"\nSuccessfully extracted metadata for {len(files_to_process)}/{len(pdf_files)} files.")
        
        # Print metadata summary
        print_metadata_summary(files_to_process)
        
        if not args.non_interactive:
            if not confirm_action("\nDo you want to proceed to text extraction and chunking?"):
                logger.info("Process cancelled by user. Exiting.")
                sys.exit(0)
    
    # STEP 3: Text Extraction, Chunking, Embedding, and FAISS Storage
    if args.skip_chunking:
        logger.info("\nSkipping text extraction and chunking is not supported without saved chunks. Exiting.")
        sys.exit(1)
    else:
        logger.info("\nSTEP 3: Text Extraction, Chunking, and Embedding")
        logger.info("This step will extract text from PDFs, chunk it, generate embeddings, and store in FAISS.")
        logger.info("This may take some time depending on the number and size of PDFs.")
        
        if not args.non_interactive:
            if not confirm_action("\nDo you want to begin processing?"):
                logger.info("Process cancelled by user. Exiting.")
                sys.exit(0)
    
    # Process each file individually to ensure correct metadata association
    success_count = 0
    for i, entry in enumerate(tqdm(files_to_process, desc="Processing files"), 1):
        logger.info(f"\nProcessing file {i}/{len(files_to_process)}: {entry['file'].name}")
        if process_report(entry["file"], entry["company"], entry["year"]):
            success_count += 1
    
    logger.info(f"\nProcessing complete. Successfully processed {success_count}/{len(files_to_process)} files.")
    
    # Final FAISS Report
    logger.info("\nGenerating final FAISS processing report...")
    print_faiss_processing_report()
    
    # Save FAISS index and metadata to disk
    logger.info("\nSTEP 4: Saving FAISS Index and Metadata")
    
    if not args.non_interactive:
        if not confirm_action("\nDo you want to save the FAISS index and metadata?"):
            logger.info("Saving cancelled by user.")
        else:
            save_faiss_index_and_metadata()
    else:
        save_faiss_index_and_metadata()
    
    logger.info("\nPROCESSING COMPLETE.")
    
    if not args.non_interactive:
        input("\nPress Enter to exit...")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user. Exiting gracefully.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
