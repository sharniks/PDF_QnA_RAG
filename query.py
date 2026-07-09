import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import ollama
from dotenv import load_dotenv
import os
import requests

# -----------------------------------------------------------------------------
# Load environment variables (.env file)
# -----------------------------------------------------------------------------
# Used for securely storing secrets like Hugging Face API token.
load_dotenv()

# Path where the FAISS index was created during the ingestion process.
INDEX_PATH = "./vectorstore/faiss_index"

# -----------------------------------------------------------------------------
# Step 1 : Load FAISS Index
# -----------------------------------------------------------------------------
# The FAISS index contains only vector embeddings.
# It does NOT contain the original text.
index = faiss.read_index(INDEX_PATH)

# -----------------------------------------------------------------------------
# Step 2 : Load Chunk Mapping
# -----------------------------------------------------------------------------
# During ingestion, we stored the original text chunks separately
# in a pickle file.
#
# Why?
# FAISS only returns vector IDs after searching.
# We need this mapping to convert those IDs back into readable text.
with open(INDEX_PATH + "_mapping.pkl", "rb") as f:
    chunks = pickle.load(f)

# -----------------------------------------------------------------------------
# Step 3 : Load Embedding Model
# -----------------------------------------------------------------------------
# IMPORTANT:
# Always use the SAME embedding model that was used while
# creating the FAISS index.
#
# If a different model is used, the embeddings will exist in a
# different vector space and similarity search will fail.
model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------------------------------------------------------
# Retrieve Relevant Context
# -----------------------------------------------------------------------------
def retrieve_context(query, k=3):
    """
    Retrieves the top-k most relevant text chunks
    for the user's question.

    Parameters
    ----------
    query : str
        User question.

    k : int
        Number of similar chunks to retrieve.

    Returns
    -------
    list
        Top-k matching text chunks.
    """

    # Convert the user query into an embedding vector.
    query_embedding = model.encode([query])

    # FAISS expects float32 vectors.
    query_embedding = np.array(query_embedding).astype("float32")

    # Search FAISS for the nearest vectors.
    #
    # Returns:
    # distances -> similarity scores
    # indices   -> IDs of matching chunks
    distances, indices = index.search(query_embedding, k)

    # Retrieve original text using chunk IDs.
    results = [chunks[i] for i in indices[0]]

    return results


# -----------------------------------------------------------------------------
# Ask Hugging Face Hosted LLM
# -----------------------------------------------------------------------------
def ask_hf(
        query,
        context,
        max_new_tokens=256,
        temperature=0.0,
        model="meta-llama/Llama-3.1-8B-Instruct:cerebras"):
    """
    Sends the retrieved context and user query to
    Hugging Face Inference API.

    Parameters
    ----------
    query : str
        User question.

    context : str
        Retrieved chunks from FAISS.

    max_new_tokens : int
        Maximum response length.

    temperature : float
        Controls randomness.
        Lower values = more deterministic answers.

    model : str
        Hugging Face hosted model.

    Returns
    -------
    str
        Generated answer.
    """

    # Read API token from environment variables.
    HF_TOKEN = os.getenv("HF_API_TOKEN")

    HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    if HF_TOKEN is None:
        raise RuntimeError(
            "HF_API_TOKEN not found. Please add it to your .env file."
        )

    # Prompt engineering:
    # We explicitly instruct the model to answer ONLY from
    # the retrieved context.
    prompt = f"""
    You are a helpful assistant.

    Answer the question using ONLY the provided context.

    If the answer is not present in the context,
    reply with "I don't know."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_new_tokens,
        "temperature": temperature
    }

    # Send request to Hugging Face API.
    response = requests.post(
        HF_API_URL,
        headers=headers,
        json=payload,
        timeout=120
    )

    # Raise exception for HTTP errors (4xx, 5xx)
    response.raise_for_status()

    # Convert JSON response into Python dictionary.
    result = response.json()

    if (
        "choices" not in result
        or not result["choices"]
        or "message" not in result["choices"][0]
        or "content" not in result["choices"][0]["message"]
    ):
        raise RuntimeError(f"Unexpected response:\n{result}")

    # Extract generated answer.
    return result["choices"][0]["message"]["content"]


# -----------------------------------------------------------------------------
# Ask Local Ollama Model
# -----------------------------------------------------------------------------
def ask_ollama(query, context):
    """
    Sends the retrieved context to a locally running
    Ollama model.

    This avoids external API calls and keeps everything local.
    """

    prompt = f"""
    You are a helpful assistant.

    Answer the question using ONLY the provided context.

    If the answer is not in the context,
    say "I don't know."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# -----------------------------------------------------------------------------
# Main Program
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    while True:

        # Read user question.
        q = input("\nAsk a question (or type 'exit'): ")

        # Exit chatbot.
        if q.lower() == "exit":
            break

        # ---------------------------------------------------------
        # Step 1 : Retrieve relevant document chunks.
        # ---------------------------------------------------------
        top_chunks = retrieve_context(q, k=3)

        # Merge retrieved chunks into one context.
        combined_context = "\n\n".join(top_chunks)

        # ---------------------------------------------------------
        # Step 2 : Send context + question to LLM.
        # ---------------------------------------------------------

        # Local LLM
        # answer = ask_ollama(q, combined_context)

        # Hugging Face Hosted LLM
        answer = ask_hf(q, combined_context)

        # ---------------------------------------------------------
        # Step 3 : Display response.
        # ---------------------------------------------------------
        print("\nAnswer:")
        print(answer)

        # ---------------------------------------------------------
        # Debugging (Optional)
        # Shows which chunks were retrieved.
        # Useful for understanding why the LLM answered a certain way.
        # ---------------------------------------------------------
        # print("\nRetrieved Chunks:\n")
        # for i, chunk in enumerate(top_chunks, start=1):
        #     print(f"{i}. {chunk[:200]}...")
