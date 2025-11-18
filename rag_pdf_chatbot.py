import re
import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="📄 PDF Chatbot – Concept Explainer", layout="wide")
st.title("📄 PDF Question-Answer Chatbot – Simple Human Explanation")


# ------------------ SIDEBAR CONTROLS ------------------
with st.sidebar:
    st.header("⚙️ QA Settings")

    top_k = st.slider("Number of chunks to search (k)", 1, 10, 5)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.3, 0.1)

    st.markdown("---")
    st.caption(
        "This app reads your PDFs, finds the most relevant parts, and then\n"
        "explains the answer in simple language using a text generation model.\n"
        "Works well for theory/ML/notes PDFs."
    )


# ------------------ EMBEDDING MODEL ------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedder = load_embedder()


# ------------------ GENERATION MODEL (FLAN-T5-BASE) ------------------
@st.cache_resource
def load_generator():
    # Use base model for speed and stability on CPU
    model_name = "google/flan-t5-base"
    return pipeline("text2text-generation", model=model_name)


generator = load_generator()


# ------------------ BUILD FAISS INDEX ------------------
@st.cache_resource
def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


# ------------------ CHAT HISTORY STATE ------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# ------------------ QUESTION TYPE DETECTION ------------------
def is_mcq_question(question: str) -> bool:
    """
    Heuristic:
    - Looks for options like: A) B) C), a), b), c)
    - Or multiple lines starting with a), b), c), d)
    """
    q = question.strip()

    # Pattern like "A) ...", "B) ...", "(A)", "(a)" etc.
    if re.search(r'\b[A-Da-d][\)\.]', q):
        return True

    # Multiple lines that look like options
    option_lines = re.findall(r'^[ \t]*[A-Da-d][\)\.]', q, flags=re.MULTILINE)
    if len(option_lines) >= 2:
        return True

    return False


def parse_mcq(question: str):
    """
    Parse MCQ question text into:
    - stem (the actual question)
    - options list: [(letter, text), ...]
    """
    lines = [l.strip() for l in question.splitlines() if l.strip()]
    stem_lines = []
    options = []

    for line in lines:
        m = re.match(r'^([A-Da-d])[\)\.\-]\s*(.*)$', line)
        if m:
            letter = m.group(1).upper()
            text = m.group(2).strip()
            options.append((letter, text))
        else:
            stem_lines.append(line)

    stem = " ".join(stem_lines).strip()
    return stem, options


def classify_question(question: str) -> str:
    """
    Returns one of: 'mcq', 'explain'
    """
    if is_mcq_question(question):
        return "mcq"
    return "explain"


# ------------------ ANSWER CLEANING HELPERS ------------------
def clean_repeated_words(text: str) -> str:
    """
    Remove ugly repetitions like:
    'back back back back' -> 'back'
    'to to to to' -> 'to'
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Collapse sequences of the same word repeated 2+ times into one
    # e.g. "back back", "back back back" -> "back"
    def collapse(match):
        w = match.group(1)
        return w

    pattern = r'\b(\w+)(?:\s+\1\b){1,}'
    text = re.sub(pattern, collapse, text, flags=re.IGNORECASE)

    return text


def clean_output_text(text: str) -> str:
    """
    General cleanup for model output:
    - collapse spaces
    - remove weird leading/trailing punctuation
    """
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip(" .-")
    return text


def trim_answer(text: str, max_chars: int = 350) -> str:
    """
    Cut the answer to a reasonable size and avoid long garbage tails.
    - If longer than max_chars, cut it.
    - Prefer cutting at the last full stop within the limit.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text

    snippet = text[:max_chars]

    # Try to cut at the last '.' inside the snippet
    last_dot = snippet.rfind('.')
    if last_dot != -1 and last_dot > max_chars * 0.3:
        return snippet[:last_dot + 1].strip()

    # Otherwise cut at last space
    if ' ' in snippet:
        snippet = snippet.rsplit(' ', 1)[0]

    return snippet.strip()


# ------------------ PDF EXTRACTION & CLEANING ------------------
uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

index = None
chunks = []
metadatas = []

if uploaded_files:
    text_pages = []  # list of (text, page_num, file_name)

    for file in uploaded_files:
        pdf_reader = PyPDF2.PdfReader(file)
        file_name = file.name

        for page_num, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text()
            if not page_text:
                continue

            clean_text = page_text.replace("\n", " ")

            # Join hyphenated words split across lines: "re- specting" -> "respecting"
            clean_text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', clean_text)

            # Remove repeated dots like ". . . . ."
            clean_text = re.sub(r'\.\s*\.+', '.', clean_text)

            # Collapse multiple spaces to single space
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()

            text_pages.append((clean_text, page_num, file_name))

    def split_text_with_metadata(text_pages, chunk_size=250, overlap=50):
        local_chunks = []
        local_metadatas = []
        for text, page_num, file_name in text_pages:
            words = text.split()
            start = 0
            while start < len(words):
                end = start + chunk_size
                chunk_words = words[start:end]
                if not chunk_words:
                    break
                chunk = " ".join(chunk_words)
                local_chunks.append(chunk)
                local_metadatas.append({"page": page_num, "file": file_name})
                start += max(1, chunk_size - overlap)
        return local_chunks, local_metadatas

    chunks, metadatas = split_text_with_metadata(text_pages)

    st.write(f"✅ Total chunks created: {len(chunks)}")

    if len(chunks) > 0:
        with st.spinner("Creating FAISS index..."):
            index = build_faiss_index(chunks)
        st.success("✅ FAISS index ready!")
    else:
        st.warning("No text could be extracted from the uploaded PDF(s).")


# ------------------ USER QUERY + CHAT HISTORY DISPLAY ------------------
user_question = st.text_input("Ask a question about your PDF(s):")

if st.session_state["chat_history"]:
    st.markdown("### 💬 Chat History")
    for msg in st.session_state["chat_history"]:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            st.markdown(f"**🧑 You:** {content}")
        else:
            st.markdown(f"**🤖 Bot:**\n```text\n{content}\n```")


# ------------------ QA LOGIC ------------------
if user_question:
    if not uploaded_files or index is None or len(chunks) == 0:
        st.error("⚠️ Please upload at least one valid PDF before asking a question.")
    else:
        q_type = classify_question(user_question)
        st.caption(f"🔎 Detected question type: **{q_type.upper()}**")

        if q_type == "mcq":
            stem, options = parse_mcq(user_question)
            if not stem or not options:
                # fallback to explanation if parsing fails
                q_type = "explain"
                stem = user_question.strip()
        else:
            stem = user_question.strip()
            options = []

        # Encode stem for retrieval
        query_emb = embedder.encode([stem], convert_to_numpy=True)

        # Search FAISS
        k = min(top_k, len(chunks))
        D, I = index.search(query_emb, k=k)

        best_dist = D[0][0]
        relevance_threshold = 2.0  # can be tuned

        if best_dist > relevance_threshold:
            msg = "The answer is not confidently available in the document."
            st.warning(msg)
            st.session_state["chat_history"].append({"role": "user", "content": user_question})
            st.session_state["chat_history"].append({"role": "assistant", "content": msg})
        else:
            retrieved_chunks = [chunks[i] for i in I[0]]
            retrieved_metas = [metadatas[i] for i in I[0]]

            context = "\n\n".join(retrieved_chunks)

            pages_used = sorted({m["page"] for m in retrieved_metas})
            files_used = sorted({m["file"] for m in retrieved_metas})
            file_list_str = ", ".join(files_used)
            page_list_str = ", ".join(map(str, pages_used))
            st.info(f"📄 Answer based on file(s): {file_list_str} | page(s): {page_list_str}")

            # ------------------ PROMPT CONSTRUCTION ------------------
            if q_type == "mcq":
                options_block = "\n".join([f"{letter}) {text}" for letter, text in options])

                prompt = (
                    "You are a helpful assistant reading a technical PDF document.\n"
                    "You are given a multiple-choice question and some relevant excerpts.\n"
                    "Use ONLY the document excerpts to choose the correct option.\n"
                    "Respond ONLY in this exact format: 'Answer: <letter>) <option_text>'.\n"
                    "Do not explain your answer.\n\n"
                    f"Document excerpts:\n{context}\n\n"
                    f"Question: {stem}\n"
                    f"Options:\n{options_block}\n\n"
                    "Answer:"
                )
            else:
                # Normal explanation mode
                prompt = (
                    "You are a friendly ML tutor. A user asked a question about a concept.\n"
                    "You are given some excerpts from a PDF related to that concept.\n"
                    "Using ONLY the information in the excerpts, explain the answer in very simple,\n"
                    "clear English so that a beginner can understand it.\n"
                    "You MUST rewrite in your own words. Do NOT copy broken phrases or repeated words\n"
                    "from the document. Avoid weird repetitions or noise from the PDF.\n"
                    "Write 3–6 sentences that directly answer the question.\n\n"
                    f"Document excerpts:\n{context}\n\n"
                    f"Question: {stem}\n\n"
                    "Answer in simple language:"
                )

            # ------------------ GENERATE ANSWER ------------------
            with st.spinner("Generating answer in simple language..."):
                try:
                    response = generator(
                        prompt,
                        max_new_tokens=220,
                        num_return_sequences=1,
                        temperature=temperature,
                        top_p=0.9,
                        do_sample=temperature > 0.0,
                        repetition_penalty=1.25,
                        no_repeat_ngram_size=4,
                    )

                    full_text = response[0]["generated_text"]
                    answer = full_text.strip()

                    # Clean and trim answer
                    answer = clean_repeated_words(answer)
                    answer = clean_output_text(answer)
                    answer = trim_answer(answer)

                    display_text = answer

                    st.success("💬 Answer:")
                    st.markdown(f"```text\n{display_text}\n```")

                    st.session_state["chat_history"].append({"role": "user", "content": user_question})
                    st.session_state["chat_history"].append({"role": "assistant", "content": display_text})

                except Exception as e:
                    st.error(f"⚠️ Error generating answer: {e}")
