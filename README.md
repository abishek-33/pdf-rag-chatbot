📄 PDF-RAG-Chatbot

A simple AI tool that answers questions from any PDF using semantic search and text generation.

🧰 Tech Stack (Super Short Summary)
Frontend

Streamlit → UI, PDF upload, question input, answer display

PDF Processing

PyPDF2 → Extracts text from PDF pages

Embeddings

SentenceTransformer (MiniLM) → Converts PDF chunks + user questions into numerical vectors

Semantic Search

FAISS → Finds the most relevant PDF chunks for the question

Text Generation

FLAN-T5-Base → Generates answers in simple human-friendly language

Handles detailed questions

Handles short answers

Handles MCQs

Cleaning

Regex (re) → Removes repeated words, fixes noisy PDF text

Performance

Streamlit caching → Speeds up model loading and FAISS indexing

⚙️ How It Works (Quick)

Upload PDF

Extract and clean text

Split into chunks

Convert chunks to embeddings

Store in FAISS index

Convert question to embedding

Retrieve top-K relevant chunks

Send context + question to FLAN-T5

Generate clean answer

Show answer with page numbers

▶️ Run
pip install -r requirements.txt
streamlit run app.py
