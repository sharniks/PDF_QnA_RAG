import fitz  # PyMuPDF library for reading PDF files
import os
import logging
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Configure logging so that INFO and ERROR messages are printed on console.
logging.basicConfig(level=logging.INFO)

# Path of the input PDF document
DATA_PATH = "./data/Sample-Java.pdf"

# Location where FAISS vector index will be stored
INDEX_PATH = "./vectorstore/faiss_index"


# -----------------------------------------------------------------------------
# Step 1 : Extract text from PDF
# -----------------------------------------------------------------------------
def extract_text_from_pdf(path):
    """
    Reads a PDF file and extracts text from every page.

    Parameters
    ----------
    path : str
        Path to the PDF document.

    Returns
    -------
    str
        Combined text from all pages.
        Returns None if the file does not exist or an error occurs.
    """

    # Verify that the PDF exists before attempting to open it.
    if os.path.exists(path):
        try:
            # Open PDF document
            pdf_document = fitz.open(path)

            # Stores text extracted from each page
            all_text = []

            # Iterate through every page in the PDF
            for page_num in range(len(pdf_document)):

                # Load one page at a time
                page = pdf_document.load_page(page_num)

                # Extract plain text from the page
                text = page.get_text()

                # Store page text
                all_text.append(text)

            # Always close the PDF after processing
            pdf_document.close()

            logging.info(
                f"PDF loaded successfully. Extracted {len(all_text)} pages."
            )

            # Merge all page texts into one large string
            return "\n".join(all_text)

        except Exception as e:
            logging.error(f"Error processing PDF: {e}")

    else:
        logging.error(f"PDF file not found at path: {path}")

    return None


# -----------------------------------------------------------------------------
# Step 2 : Split text into smaller chunks
# -----------------------------------------------------------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits a large text into overlapping chunks.

    Why overlap?
    ------------
    Overlap preserves context between consecutive chunks.
    Without overlap, important information at chunk boundaries
    may be lost.

    Example:
        Chunk 1 : ABCDEFGHIJ
        Chunk 2 : HIJKLMNOPQ

    Parameters
    ----------
    text : str
        Complete extracted text.

    chunk_size : int
        Maximum characters in one chunk.

    overlap : int
        Number of characters shared between adjacent chunks.

    Returns
    -------
    list
        List of text chunks.
    """

    chunks = []
    start = 0

    while start < len(text):

        # End position of current chunk
        end = start + chunk_size

        # Extract the chunk
        chunks.append(text[start:end])

        # Move start position while keeping overlap
        start += chunk_size - overlap

    return chunks


# -----------------------------------------------------------------------------
# Step 3 : Generate embeddings
# -----------------------------------------------------------------------------
def create_embeddings(chunks):
    """
    Converts text chunks into numerical vectors (embeddings).

    These vectors capture semantic meaning so that similar texts
    are located close together in vector space.

    Model used:
        all-MiniLM-L6-v2

    Returns
    -------
    numpy.ndarray
        Embedding vector for each chunk.
    """

    # Load SentenceTransformer model
    # (Downloads automatically the first time)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embedding for every chunk
    vectors = model.encode(
        chunks,
        show_progress_bar=True
    )

    return vectors


# -----------------------------------------------------------------------------
# Step 4 : Save embeddings into FAISS
# -----------------------------------------------------------------------------
def save_faiss_index(embeddings, chunks, index_path=INDEX_PATH):
    """
    Creates a FAISS index from embeddings and saves it to disk.

    Also stores a mapping between vector index and original text
    so retrieved vectors can be converted back into readable text.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Embedding vectors.

    chunks : list
        Original text chunks.

    index_path : str
        Destination path for FAISS index.
    """

    # Embedding dimension (e.g. 384 for MiniLM)
    dim = embeddings.shape[1]

    # Create a simple FAISS index using L2 (Euclidean) distance.
    # This index performs exact nearest-neighbor search.
    index = faiss.IndexFlatL2(dim)

    # Add all embedding vectors into the index
    index.add(embeddings)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    # Save FAISS index to disk
    faiss.write_index(index, index_path)

    # Save original text chunks separately.
    # FAISS stores vectors only, not the original text.
    with open(index_path + "_mapping.pkl", "wb") as f:
        pickle.dump(chunks, f)

    logging.info(f"FAISS index saved at {index_path}")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Step 1 : Read PDF
    text = extract_text_from_pdf(DATA_PATH)

    # Stop execution if PDF extraction failed
    if text is None:
        raise ValueError("Failed to extract text from PDF.")

    # Step 2 : Split into chunks
    chunks = chunk_text(text)

    logging.info(f"Total chunks created: {len(chunks)}")

    # Step 3 : Convert chunks into embeddings
    embeddings = create_embeddings(chunks)

    logging.info(f"Embedding shape: {embeddings.shape}")

    # Step 4 : Store vectors in FAISS
    save_faiss_index(np.array(embeddings), chunks)

    logging.info("Vector database created successfully.")