# 📄 PDF RAG Chat

A simple Retrieval-Augmented Generation (RAG) application that allows you to ask questions about a PDF document from the terminal.

The application extracts text from a PDF, generates embeddings, stores them in a FAISS vector database, and retrieves relevant context to answer user queries.

This project was built as part of my AI Engineering learning journey to understand the fundamentals of Retrieval-Augmented Generation (RAG).

---

## ✨ Features

- 📄 Load PDF documents
- ✂️ Automatic text chunking
- 🧠 Generate embeddings
- 🗂️ Store embeddings using FAISS
- 💬 Ask questions about the document
- 💻 Simple terminal-based interface

---

## 🛠️ Tech Stack

- Python
- LangChain
- Hugging Face
- FAISS
- Sentence Transformers / Embedding Model
- Python Dotenv

---

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/sharniks/pdf-rag-chat.git
cd pdf-rag-chat
```

### Create a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment variables

Create a `.env` file in the project root.

```text
HF_API_TOKEN=your_huggingface_api_token
```

### Add your PDF

Place the PDF inside the `data/` directory.

Example:

```
data/
└── sample.pdf
```

### Generate embeddings

```bash
python ingest.py
```

This will:

- Extract text from the PDF
- Split text into chunks
- Generate embeddings
- Create and save the FAISS vector index

### Start the application

```bash
python query.py
```

Example:

```text
Ask a question (or type 'exit'):

> What is polymorphism?

Answer:
...
```

Type `exit` to quit.

---

## 📂 Project Structure

```
pdf-rag-chat/
│
├── data/
├── faiss_index/
├── ingest.py
├── query.py
├── requirements.txt
└── README.md
```

---

## 📚 Concepts Practiced

- Retrieval-Augmented Generation (RAG)
- Document Loading
- Text Chunking
- Embeddings
- Vector Databases (FAISS)
- Semantic Search
- Prompt Engineering

---

## 🚀 Future Improvements

- Streamlit Web UI
- Multiple PDF Support
- Chat History
- Source Citations
- Hybrid Search
- Reranking
- Local LLM Support (Ollama)

---

## 👨‍💻 Author

**Nikhil**

Software Engineer transitioning into AI Engineering.
Building production-ready AI applications and sharing my learning journey through hands-on projects.
