# PDF_QnA_RAG
Practice Python Project for RAG

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/sharniks/PDF_QnA_RAG.git
cd PDF_QnA_RAG
```

### 2. Create and activate a virtual environment (recommended)

Creating a virtual environment keeps the project's dependencies isolated from your system Python.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

**Windows (Command Prompt)**

```cmd
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root and add your Hugging Face API token:

```text
HF_API_TOKEN=your_huggingface_api_token
```

### 5. Add your PDF

Place the PDF you want to query inside the `data/` folder.

Example:

```
data/
└── Sample-Java.pdf
```

### 6. Generate embeddings and create the FAISS index

Run the ingestion script:

```bash
python ingest.py
```

This will:

- Extract text from the PDF
- Split it into chunks
- Generate embeddings
- Create and save the FAISS index

### 7. Start the chatbot

Once the FAISS index has been created, run:

```bash
python query.py
```

Ask questions about your PDF:

```text
Ask a question (or type 'exit'):
```

Type `exit` to quit.
