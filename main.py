import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import tempfile

st.set_page_config(page_title="PDF to Chatbot", page_icon="🤖", layout="centered")

st.markdown("""
<style>
/* ── Background ── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #6b48ff 0%, #a855f7 60%, #7c3aed 100%) !important;
    min-height: 100vh;
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
#MainMenu, footer         { display: none !important; }

/* ── Main block = the white card ── */
.main .block-container {
    background: #ffffff;
    border-radius: 22px;
    padding: 44px 44px 36px 44px !important;
    max-width: 500px !important;
    margin: 64px auto 0 auto !important;
    box-shadow: 0 24px 64px rgba(0,0,0,0.28);
}

/* ── Hero ── */
.hero { text-align: center; margin-bottom: 28px; }
.hero .icon  { font-size: 2.6rem; line-height: 1; margin-bottom: 10px; }
.hero h1 {
    font-size: 1.85rem; font-weight: 800;
    color: #111827; margin: 0 0 8px 0;
}
.hero p { font-size: 0.93rem; color: #6b7280; margin: 0; }

/* ── Upload label ── */
.up-label {
    font-size: 0.82rem; font-weight: 700;
    color: #374151; margin-bottom: 6px;
}

/* ── File uploader box ── */
[data-testid="stFileUploader"] section {
    background: #f9fafb !important;
    border: 2px dashed #d1d5db !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #7c3aed !important;
}
[data-testid="stFileUploader"] label { display: none !important; }

/* ── Button ── */
div.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #6d28d9) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 11px !important;
    padding: 13px 0 !important;
    width: 100% !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    margin-top: 14px !important;
    transition: opacity .2s;
}
div.stButton > button:hover { opacity: 0.85 !important; }

/* ── Powered-by ── */
.powered {
    text-align: center; font-size: 0.78rem;
    color: #9ca3af; margin-top: 16px;
}

/* ── Tags below card ── */
.tags {
    text-align: center; font-size: 0.78rem;
    color: rgba(255,255,255,0.75);
    margin-top: 22px; letter-spacing: 0.04em;
}

/* ── Chat bubbles ── */
.bubble-user {
    background: linear-gradient(90deg,#7c3aed,#6d28d9);
    color: #fff; border-radius: 18px 18px 4px 18px;
    padding: 11px 15px; margin: 6px 0 6px 60px;
    font-size: 0.91rem; line-height: 1.5;
}
.bubble-bot {
    background: #f3f4f6; color: #111827;
    border-radius: 18px 18px 18px 4px;
    padding: 11px 15px; margin: 6px 60px 6px 0;
    font-size: 0.91rem; line-height: 1.5;
}

/* ── Chat text input ── */
[data-testid="stTextInput"] input {
    border: 2px solid #e5e7eb !important;
    border-radius: 10px !important;
    font-size: 0.92rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #7c3aed !important;
    box-shadow: none !important;
}

/* ── "Upload a new PDF" reset button — make it subtle ── */
div.stButton > button[kind="secondary"],
div.stButton:last-of-type > button {
    background: transparent !important;
    color: #6b7280 !important;
    border: 1.5px solid #d1d5db !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 8px 0 !important;
    margin-top: 4px !important;
}
div.stButton:last-of-type > button:hover {
    border-color: #7c3aed !important;
    color: #7c3aed !important;
    opacity: 1 !important;
}

/* ── Alerts / info ── */
[data-testid="stAlert"] { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ── API key ────────────────────────────────────────────────────────────────────
def get_groq_api_key():
    key = os.getenv('GROQ_API_KEY')
    if key:
        return key
    for path in ["groq_api.txt", "./groq_api.txt", os.path.join(os.getcwd(), "groq_api.txt")]:
        try:
            with open(path) as f:
                k = f.read().strip()
                if k:
                    return k
        except FileNotFoundError:
            continue
    st.error("Groq API key not found. Set GROQ_API_KEY or create groq_api.txt.")
    st.stop()

groq_api_key = get_groq_api_key()

# ── Session state ──────────────────────────────────────────────────────────────
for k, default in [('vectorstore', None), ('llm', None), ('chat_history', []), ('input_key', 0)]:
    if k not in st.session_state:
        st.session_state[k] = default

# ══════════════════════════════════════════════════════════════════════════════
#  LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.vectorstore is None:

    # Hero header
    st.markdown("""
    <div class="hero">
        <div class="icon">🤖</div>
        <h1>PDF to Chatbot</h1>
        <p>Turn any PDF into an intelligent AI chatbot in seconds</p>
    </div>
    """, unsafe_allow_html=True)

    # Upload label + widget
    st.markdown('<p class="up-label">Enter your PDF</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

    # CTA button
    start = st.button("🚀  Start Building Chatbot")

    # Powered-by
    st.markdown('<p class="powered">— Powered by Groq, LangChain & FAISS —</p>', unsafe_allow_html=True)

    # Tags row (outside the card, on the gradient)
    st.markdown(
        '<p class="tags">RAG &nbsp;·&nbsp; Vector Search &nbsp;·&nbsp; Semantic Retrieval &nbsp;·&nbsp; Groq LLM</p>',
        unsafe_allow_html=True,
    )

    # Process
    if start:
        if not uploaded_file:
            st.warning("Please upload a PDF first.")
        else:
            with st.spinner("Processing your PDF…"):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    docs = RecursiveCharacterTextSplitter(
                        chunk_size=500, chunk_overlap=50
                    ).split_documents(PyPDFLoader(tmp_path).load())

                    vs = FAISS.from_documents(
                        docs, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    )
                    llm = ChatGroq(model="openai/gpt-oss-120b", api_key=groq_api_key, temperature=0)

                    st.session_state.vectorstore = vs
                    st.session_state.llm = llm
                    st.session_state.chat_history = []
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                    st.rerun()

                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

# ══════════════════════════════════════════════════════════════════════════════
#  CHAT PAGE
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div style="text-align:center; margin-bottom:24px;">
        <span style="font-size:1.5rem; font-weight:800; color:#111827;">🤖 PDF Chatbot</span><br>
        <span style="font-size:0.85rem; color:#6b7280;">Ask anything about your document</span>
    </div>
    """, unsafe_allow_html=True)

    # Chat history bubbles
    if not st.session_state.chat_history:
        st.markdown(
            '<div class="bubble-bot">Hi! Your PDF is ready. Ask me anything about it.</div>',
            unsafe_allow_html=True,
        )

    for msg in st.session_state.chat_history:
        cls = "bubble-user" if msg["role"] == "user" else "bubble-bot"
        st.markdown(f'<div class="{cls}">{msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

    # Input row — use input_key to reset the field after each send
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input(
            "", placeholder="Ask a question about your document…",
            label_visibility="collapsed",
            key=f"q_{st.session_state.input_key}"
        )
    with col2:
        send = st.button("Send")

    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    if st.button("↩ Upload a new PDF"):
        st.session_state.vectorstore = None
        st.session_state.llm = None
        st.session_state.chat_history = []
        st.session_state.input_key = 0
        st.rerun()

    # Generate answer and clear input
    if send and question:
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.spinner("Thinking…"):
            try:
                relevant_docs = st.session_state.vectorstore.similarity_search(question, k=4)
                context = "\n\n".join(d.page_content for d in relevant_docs)
                prompt = f"""Use the following context from the document to answer the question.
If you don't know the answer based on the context, say "I don't know."
Keep your answer concise and based only on the provided context.

Context:
{context}

Question: {question}

Answer:"""
                resp = st.session_state.llm.invoke(prompt)
                answer = resp.content if hasattr(resp, "content") else str(resp)
            except Exception as e:
                answer = f"Error: {e}"

        st.session_state.chat_history.append({"role": "bot", "content": answer})
        st.session_state.input_key += 1   # clears the text input on rerun
        st.rerun()
