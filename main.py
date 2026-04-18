import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile

st.set_page_config(page_title="PDF Chatbot", page_icon="📖", layout="wide")


def get_gemini_api_key():
    key = os.getenv('GOOGLE_API_KEY')
    if key:
        return key
    for path in ["gemini_api.txt", "./gemini_api.txt", os.path.join(os.getcwd(), "gemini_api.txt")]:
        try:
            with open(path) as f:
                k = f.read().strip()
                if k:
                    return k
        except FileNotFoundError:
            continue
    st.error("Gemini API key not found. Set GOOGLE_API_KEY or create gemini_api.txt.")
    st.stop()


gemini_api_key = get_gemini_api_key()

for k, default in [('vectorstore', None), ('llm', None), ('chat_history', []), ('input_key', 0)]:
    if k not in st.session_state:
        st.session_state[k] = default

nav_page = st.query_params.get("page", "home")

SHARED_STYLE = (
    "<style>"
    "[data-testid='stAppViewContainer']{background:linear-gradient(135deg,#ede9fe 0%,#f8f7ff 50%,#ede9fe 100%) !important;min-height:100vh;}"
    "[data-testid='stHeader'],[data-testid='stToolbar'],[data-testid='stDecoration']{display:none !important;}"
    "#MainMenu,footer{display:none !important;}"
    ".main .block-container{background:#ffffff;border-radius:24px;max-width:1160px !important;padding:0 0 60px 0 !important;margin:28px auto 40px auto !important;box-shadow:0 4px 48px rgba(109,40,217,0.10);overflow:hidden;}"
    ".stVerticalBlock{gap:0 !important;}"
    ".element-container{margin-bottom:0 !important;}"
    "[data-testid='stHorizontalBlock']{gap:0 !important;align-items:stretch !important;}"
    "[data-testid='stAlert']{border-radius:10px !important;}"
    "</style>"
)


def render_navbar(active=""):
    feat_style = "color:#7c3aed;font-weight:700;" if active == "features" else "color:#374151;font-weight:500;"
    return f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
                padding:22px 52px;border-bottom:1px solid #f3f4f6;">
        <a href="/" style="display:flex;align-items:center;gap:10px;
                    font-size:1.1rem;font-weight:700;color:#3b0764;text-decoration:none;">
            <span style="font-size:1.45rem;">📖</span> PDF Chatbot
        </a>
        <div style="display:flex;align-items:center;gap:30px;">
            <a href="/?page=features" style="{feat_style}font-size:.9rem;text-decoration:none;">Features</a>
            <a href="#" style="color:#374151;font-size:.9rem;font-weight:500;text-decoration:none;">Pricing</a>
            <a href="#" style="color:#374151;font-size:.9rem;font-weight:500;text-decoration:none;">API</a>
            <a href="#" style="color:#374151;font-size:.9rem;font-weight:500;text-decoration:none;">Sign In</a>
            <a href="/" style="background:#1e1b4b;color:#fff;padding:10px 22px;
                               border-radius:50px;font-weight:600;font-size:.88rem;
                               text-decoration:none;">Get Started Free</a>
        </div>
    </div>
    """


# ══════════════════════════════════════════════════════════════════════════════
#  CHAT PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.vectorstore is not None:
    st.markdown(
        "<style>"
        "[data-testid='stAppViewContainer']{background:linear-gradient(135deg,#ede9fe 0%,#f8f7ff 50%,#ede9fe 100%) !important;min-height:100vh;}"
        "[data-testid='stHeader'],[data-testid='stToolbar'],[data-testid='stDecoration']{display:none !important;}"
        "#MainMenu,footer{display:none !important;}"
        ".main .block-container{background:#ffffff;border-radius:22px;max-width:700px !important;padding:44px 44px 36px 44px !important;margin:64px auto 0 auto !important;box-shadow:0 24px 64px rgba(0,0,0,.18);}"
        ".bubble-user{background:linear-gradient(90deg,#7c3aed,#6d28d9);color:#fff;border-radius:18px 18px 4px 18px;padding:11px 15px;margin:6px 0 6px 60px;font-size:.91rem;line-height:1.5;}"
        ".bubble-bot{background:#f3f4f6;color:#111827;border-radius:18px 18px 18px 4px;padding:11px 15px;margin:6px 60px 6px 0;font-size:.91rem;line-height:1.5;}"
        "[data-testid='stTextInput'] input{border:2px solid #e5e7eb !important;border-radius:10px !important;font-size:.92rem !important;}"
        "[data-testid='stTextInput'] input:focus{border-color:#7c3aed !important;box-shadow:none !important;}"
        "div.stButton > button{background:linear-gradient(90deg,#7c3aed,#6d28d9) !important;color:#fff !important;border:none !important;border-radius:11px !important;padding:13px 0 !important;width:100% !important;font-size:1rem !important;font-weight:700 !important;margin-top:14px !important;transition:opacity .2s;}"
        "div.stButton > button:hover{opacity:.85 !important;}"
        "div.stButton:last-of-type > button{background:transparent !important;color:#6b7280 !important;border:1.5px solid #d1d5db !important;font-size:.85rem !important;font-weight:500 !important;padding:8px 0 !important;margin-top:4px !important;}"
        "div.stButton:last-of-type > button:hover{border-color:#7c3aed !important;color:#7c3aed !important;opacity:1 !important;}"
        "[data-testid='stAlert']{border-radius:10px !important;}"
        "</style>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div style='text-align:center;margin-bottom:24px;'>"
        "<span style='font-size:1.5rem;font-weight:800;color:#111827;'>🤖 PDF Chatbot</span><br>"
        "<span style='font-size:.85rem;color:#6b7280;'>Ask anything about your document</span>"
        "</div>",
        unsafe_allow_html=True
    )

    if not st.session_state.chat_history:
        st.markdown('<div class="bubble-bot">Hi! Your PDF is ready. Ask me anything about it.</div>',
                    unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        cls = "bubble-user" if msg["role"] == "user" else "bubble-bot"
        st.markdown(f'<div class="{cls}">{msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

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
        st.session_state.input_key += 1
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURES PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif nav_page == "features":
    st.markdown(SHARED_STYLE, unsafe_allow_html=True)
    st.markdown(render_navbar("features"), unsafe_allow_html=True)

    st.markdown(
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:0;align-items:center;padding:60px 52px 70px 52px;'>"
        "<div>"
        "<h1 style='font-size:2.8rem;font-weight:900;color:#111827;line-height:1.1;margin:0 0 14px 0;'>Core Capabilities.</h1>"
        "<p style='font-size:1rem;color:#6b7280;margin:0 0 36px 0;'>Everything you need to extract insights from your documents.</p>"
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;'>"
        "<div style='background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:24px 20px;box-shadow:0 2px 10px rgba(0,0,0,.05);'>"
        "<div style='font-size:2rem;margin-bottom:14px;'>📚</div>"
        "<h3 style='font-size:1rem;font-weight:700;color:#111827;margin:0 0 8px 0;'>Context-Aware<br>Summaries</h3>"
        "<p style='font-size:.84rem;color:#6b7280;margin:0;line-height:1.55;'>Generate accurate summaries of any document length.</p>"
        "</div>"
        "<div style='background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:24px 20px;box-shadow:0 2px 10px rgba(0,0,0,.05);'>"
        "<div style='font-size:2rem;margin-bottom:14px;'>📂</div>"
        "<h3 style='font-size:1rem;font-weight:700;color:#111827;margin:0 0 8px 0;'>Cross-Document<br>Querying</h3>"
        "<p style='font-size:.84rem;color:#6b7280;margin:0;line-height:1.55;'>Analyze and find information across multiple documents.</p>"
        "</div>"
        "<div style='background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:24px 20px;box-shadow:0 2px 10px rgba(0,0,0,.05);'>"
        "<div style='font-size:2rem;margin-bottom:14px;'>🌐</div>"
        "<h3 style='font-size:1rem;font-weight:700;color:#111827;margin:0 0 8px 0;'>Multilingual Support</h3>"
        "<p style='font-size:.84rem;color:#6b7280;margin:0;line-height:1.55;'>Communicate and analyse documents in over 20 languages.</p>"
        "</div>"
        "<div style='background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:24px 20px;box-shadow:0 2px 10px rgba(0,0,0,.05);'>"
        "<div style='font-size:2rem;margin-bottom:14px;'>📖</div>"
        "<h3 style='font-size:1rem;font-weight:700;color:#111827;margin:0 0 8px 0;'>Citation &amp; Reference</h3>"
        "<p style='font-size:.84rem;color:#6b7280;margin:0;line-height:1.55;'>Automatic page and caption references for all answers.</p>"
        "</div>"
        "</div>"
        "</div>"
        "<div style='display:flex;align-items:center;justify-content:center;'>"
        "<div style='position:relative;width:420px;height:420px;'>"
        "<div style='position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:340px;height:340px;background:radial-gradient(circle,#ede9fe 0%,transparent 68%);border-radius:50%;'></div>"
        "<svg style='position:absolute;top:0;left:0;width:100%;height:100%;z-index:1;' viewBox='0 0 420 420'>"
        "<defs><marker id='arr' markerWidth='8' markerHeight='8' refX='6' refY='3' orient='auto'><path d='M0,0 L0,6 L8,3 z' fill='#a78bfa'/></marker></defs>"
        "<path d='M 105 105 Q 155 175 195 205' stroke='#c4b5fd' stroke-width='2' fill='none' marker-end='url(#arr)'/>"
        "<path d='M 318 100 Q 275 170 235 200' stroke='#c4b5fd' stroke-width='2' fill='none' marker-end='url(#arr)'/>"
        "<path d='M 100 318 Q 155 275 195 235' stroke='#c4b5fd' stroke-width='2' fill='none' marker-end='url(#arr)'/>"
        "<path d='M 248 228 Q 295 275 318 305' stroke='#c4b5fd' stroke-width='2' fill='none' marker-end='url(#arr)'/>"
        "</svg>"
        "<div style='position:absolute;top:20px;left:20px;z-index:2;background:#fff;border-radius:50%;width:80px;height:80px;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 18px rgba(109,40,217,.18);font-size:2.2rem;'>📚</div>"
        "<div style='position:absolute;top:20px;right:20px;z-index:2;background:#fff;border-radius:50%;width:80px;height:80px;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 18px rgba(109,40,217,.18);font-size:2.2rem;'>📂</div>"
        "<div style='position:absolute;top:50%;left:46%;transform:translate(-50%,-50%);z-index:3;'>"
        "<div style='background:#fff;border-radius:14px;padding:18px 16px;box-shadow:0 8px 36px rgba(109,40,217,.18);width:116px;'>"
        "<div style='width:74px;height:9px;background:#e5e7eb;border-radius:4px;margin:0 auto 7px;'></div>"
        "<div style='width:74px;height:9px;background:#e5e7eb;border-radius:4px;margin:0 auto 7px;'></div>"
        "<div style='width:74px;height:9px;background:#e5e7eb;border-radius:4px;margin:0 auto 7px;'></div>"
        "<div style='width:52px;height:9px;background:#e5e7eb;border-radius:4px;margin:0 auto;'></div>"
        "</div>"
        "<div style='text-align:center;font-size:.71rem;color:#374151;margin-top:8px;font-weight:500;'>Annual_Report.pdf</div>"
        "</div>"
        "<div style='position:absolute;bottom:40px;left:20px;z-index:2;background:#fff;border-radius:50%;width:80px;height:80px;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 18px rgba(109,40,217,.18);font-size:2.2rem;'>🌐</div>"
        "<div style='position:absolute;bottom:24px;right:6px;z-index:3;'>"
        "<div style='background:#f3f0ff;border:1.5px solid #c4b5fd;border-radius:16px;padding:16px 18px;width:138px;text-align:center;box-shadow:0 4px 22px rgba(109,40,217,.2);'>"
        "<div style='font-size:1.9rem;margin-bottom:8px;'>📖</div>"
        "<div style='font-size:.72rem;font-weight:700;color:#4c1d95;line-height:1.4;'>Citation &amp;<br>Reference Engine</div>"
        "</div>"
        "</div>"
        "</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
#  LANDING PAGE
# ══════════════════════════════════════════════════════════════════════════════
else:
    LANDING_STYLE = (
        "<style>"
        "[data-testid='stHorizontalBlock'] > [data-testid='column']:first-child {padding:52px 24px 44px 52px !important;}"
        "[data-testid='stHorizontalBlock'] > [data-testid='column']:last-child {padding:32px 52px 32px 24px !important;display:flex !important;align-items:center !important;justify-content:center !important;}"
        "[data-testid='stFileUploader'] > label {display:none !important;}"
        "[data-testid='stFileUploader'] section {background:#faf8ff !important;border:2px dashed #c4b5fd !important;border-radius:12px !important;margin-top:20px;}"
        "[data-testid='stFileUploader'] section:hover {border-color:#7c3aed !important;}"
        "div.stButton > button {background:#1e1b4b !important;color:#fff !important;border:none !important;border-radius:50px !important;padding:14px 30px !important;font-size:1rem !important;font-weight:700 !important;margin-top:16px !important;transition:opacity .2s;}"
        "div.stButton > button:hover {opacity:.88 !important;background:#1e1b4b !important;color:#fff !important;}"
        "</style>"
    )
    st.markdown(SHARED_STYLE + LANDING_STYLE, unsafe_allow_html=True)

    st.markdown(render_navbar(), unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(
            "<h1 style='font-size:2.55rem;font-weight:900;color:#111827;line-height:1.15;margin:0 0 18px 0;'>Chat With Your PDFs.<br>Instantly Get Answers.</h1>"
            "<p style='font-size:.97rem;color:#6b7280;line-height:1.65;margin:0 0 26px 0;max-width:440px;'>Tired of reading hundreds of pages? Upload any PDF and ask your questions. It's like having a search engine for your documents.</p>"
            "<p style='font-size:.82rem;font-weight:700;color:#374151;margin:0 0 6px 0;'>Upload your PDF to get started:</p>",
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
        start = st.button("👉  Start Chatting – Free!")

    with col_r:
        st.markdown(
            "<div style='position:relative;display:flex;align-items:center;justify-content:center;min-height:320px;width:100%;'>"
            "<div style='position:absolute;width:310px;height:310px;background:radial-gradient(circle,#ddd6fe 0%,transparent 70%);border-radius:50%;'></div>"
            "<div style='position:relative;z-index:1;background:#fff;border-radius:18px;box-shadow:0 16px 48px rgba(109,40,217,.18);overflow:hidden;width:340px;'>"
            "<div style='background:#4c1d95;padding:10px 14px;display:flex;gap:6px;align-items:center;'>"
            "<div style='width:10px;height:10px;border-radius:50%;background:#ff5f57;'></div>"
            "<div style='width:10px;height:10px;border-radius:50%;background:#febc2e;'></div>"
            "<div style='width:10px;height:10px;border-radius:50%;background:#28c840;'></div>"
            "</div>"
            "<div style='padding:18px;'>"
            "<div style='background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:5px 11px;font-size:.73rem;color:#374151;display:inline-block;margin-bottom:14px;'>📄 Annual_Report.pdf</div>"
            "<div style='background:#7c3aed;color:#fff;border-radius:12px 12px 4px 12px;padding:9px 13px;font-size:.78rem;margin:0 0 10px 50px;line-height:1.4;'>User: \"What was the net profit?\"</div>"
            "<div style='background:#f3f4f6;color:#111827;border-radius:12px 12px 12px 4px;padding:9px 13px;font-size:.78rem;margin:0 20px 0 0;line-height:1.4;'>AI: \"Net profit was $12.5M, a 15% increase. (Page 12)\"</div>"
            "<div style='display:flex;justify-content:flex-end;margin-top:12px;'>"
            "<div style='background:#ede9fe;border-radius:50%;width:30px;height:30px;display:flex;align-items:center;justify-content:center;font-size:.85rem;color:#7c3aed;'>&#10148;</div>"
            "</div>"
            "</div>"
            "</div>"
            "<div style='position:absolute;bottom:14px;left:14px;font-size:1.8rem;'>📚</div>"
            "<div style='position:absolute;top:14px;right:4px;background:#7c3aed;color:#fff;border-radius:50%;width:34px;height:34px;display:flex;align-items:center;justify-content:center;font-size:.9rem;box-shadow:0 4px 12px rgba(124,58,237,.4);'>💬</div>"
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown(
        "<div style='padding:60px 52px 50px 52px;background:#fafbff;border-top:1px solid #f0f0ff;border-radius:25px;margin-top:20px;'>"
        "<div style='text-align:center;font-size:1.6rem;font-weight:800;color:#111827;margin-bottom:6px;'>Key Features</div>"
        "<div style='text-align:center;font-size:.88rem;color:#9ca3af;margin-bottom:36px;'>How It Works</div>"
        "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:20px;'>"
        "<div style='background:#fff;border:1px solid #ede9fe;border-radius:16px;padding:24px 20px;box-shadow:0 2px 12px rgba(109,40,217,.06);'>"
        "<h3 style='font-size:1.02rem;font-weight:700;color:#111827;margin:0 0 10px 0;'>📁 Drag &amp; Drop</h3>"
        "<p style='font-size:.86rem;color:#6b7280;line-height:1.6;margin:0;'>Simply upload your PDF, contract, or academic paper.</p>"
        "</div>"
        "<div style='background:#fff;border:1px solid #ede9fe;border-radius:16px;padding:24px 20px;box-shadow:0 2px 12px rgba(109,40,217,.06);'>"
        "<h3 style='font-size:1.02rem;font-weight:700;color:#111827;margin:0 0 10px 0;'>🧠 Instant Insight</h3>"
        "<p style='font-size:.86rem;color:#6b7280;line-height:1.6;margin:0;'>Our AI reads and understands your entire document in seconds.</p>"
        "</div>"
        "<div style='background:#fff;border:1px solid #ede9fe;border-radius:16px;padding:18px 16px;box-shadow:0 2px 12px rgba(109,40,217,.06);'>"
        "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>"
        "<span style='font-size:1rem;'>💬</span>"
        "<span><span style='font-size:.76rem;color:#374151;font-weight:500;'>Preview: &quot;Contract_Draft_v3.pdf&quot;</span>"
        "<span style='font-size:.73rem;color:#7c3aed;font-weight:500;margin-left:6px;'>[Change File]</span></span>"
        "</div>"
        "<div style='background:#7c3aed;color:#fff;border-radius:10px 10px 3px 10px;padding:6px 10px;font-size:.73rem;margin:0 0 6px 30px;'>You: Is there a termination clause?</div>"
        "<div style='background:#f3f4f6;color:#111827;border-radius:10px 10px 10px 3px;padding:6px 10px;font-size:.73rem;margin:0 5px 8px 0;line-height:1.5;'>AI: Yes, on Page 19. It requires 90 days&#39; written notice.</div>"
        "<div style='border:1px solid #e5e7eb;border-radius:8px;padding:7px 10px;font-size:.73rem;color:#9ca3af;display:flex;justify-content:space-between;align-items:center;'>"
        "<span>Type your question here...</span><span style='color:#7c3aed;'>&#10148;</span>"
        "</div>"
        "</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='padding:50px 52px 40px 52px;text-align:center;border-top:1px solid #f0f0ff;'>"
        "<div style='font-size:1.6rem;font-weight:800;color:#111827;margin-bottom:6px;'>Try it Now</div>"
        "<div style='font-size:.88rem;color:#9ca3af;'>Demo Section</div>"
        "</div>",
        unsafe_allow_html=True,
    )

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
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=gemini_api_key, temperature=0)

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
