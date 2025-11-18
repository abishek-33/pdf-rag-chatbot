📄 PDF-RAG-Chatbot

An intelligent PDF Question-Answering system built using a Retrieval-Augmented Generation (RAG) pipeline.
The application allows users to upload any PDF, ask questions, and receive clear, concise, human-friendly answers based entirely on the document content.

🚀 Tech Stack
1. Streamlit (Frontend & App Framework)

Builds the web interface

Handles PDF upload, question input, sidebar controls

Displays the final generated answers

2. PyPDF2 (PDF Text Extraction)

Reads PDF files page-by-page

Extracts raw text while removing line breaks, hyphens, and formatting noise

3. SentenceTransformer – MiniLM (Embeddings)

Model: all-MiniLM-L6-v2

Converts text chunks and user questions into vector embeddings

Enables semantic understanding of the document

4. FAISS (Semantic Vector Search)

High-performance similarity search library

Finds the most relevant chunks from the PDF based on the user’s question

Ensures fast retrieval even for large documents

5. FLAN-T5-Base (Answer Generation)

Lightweight text-generation model from HuggingFace

Produces clear and simple explanations using the retrieved PDF text

Supports:

Conceptual answers

Short fact-based answers

MCQ option selection

6. Regex / Text Cleaning

Removes noisy patterns created by PDF formatting

Fixes repeated words and broken sentences

Ensures clean and readable output

7. Streamlit Caching (@st.cache_resource)

Caches embedding models, FAISS index, and generator

Makes the app smooth and responsive during repeated queries

🧠 How the System Works

User uploads one or more PDF files

Raw text is extracted and cleaned

Text is split into overlapping chunks

Each chunk is converted to a vector embedding (MiniLM)

FAISS stores all chunk embeddings

User question → converted to embedding

FAISS retrieves the top-K most relevant chunks

A structured prompt is generated

FLAN-T5 generates a clean, beginner-friendly answer

Output is cleaned and displayed with file & page references

▶️ Run the App
pip install -r requirements.txt
streamlit run app.py

📘 Use Cases

Understanding ML/AI lecture notes

Summarizing technical PDFs

Extracting answers from exam PDFs

Explaining concepts from research papers

MCQ identification and problem-solving

👨‍💻 Author

Abishek Sharma
GitHub: abishek-33
