import os
import streamlit as st
import pypdf
import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

def _get_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""

GEMINI_API_KEY = _get_api_key()
if not GEMINI_API_KEY:
    st.error("Thiếu GEMINI_API_KEY. Hãy thêm vào .streamlit/secrets.toml (local) hoặc Settings → Secrets (Streamlit Cloud).")
    st.stop()

genai_client = genai.Client(api_key=GEMINI_API_KEY)
client = chromadb.Client()
LLM_MODEL = "gemini-flash-latest"
EMBED_MODEL = "gemini-embedding-001"
PROMPT = """Bạn là trợ lý hỏi đáp. Dùng các đoạn ngữ cảnh dưới đây để trả lời câu hỏi.
Nếu ngữ cảnh không có thông tin, hãy nói là bạn không biết, đừng bịa.
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""

for k, v in {"collection": None, "pdf_names": [], "chat_history": [], "suggested_questions": []}.items():
    st.session_state.setdefault(k, v)

def embed(texts, task_type="RETRIEVAL_DOCUMENT"):
    resp = genai_client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in resp.embeddings]

def chunk_text(text, size=1000, overlap=200):
    paras = [p.strip() for p in text.split("\n") if p.strip()] # page 1 \n page 2
    chunks, cur = [], ""
    for p in paras:
        # Nếu một đoạn dài hơn size, cắt nhỏ đoạn đó (vẫn giữ overlap)
        while len(p) > size: # pages 1: 2000 => 1000 (-200) => 800
            if cur:
                chunks.append(cur.strip())
                cur = ""
            chunks.append(p[:size].strip())
            p = p[size - overlap:]
        if len(cur) + len(p) + 1 <= size:
            cur += p + "\n"
        else:
            if cur:
                chunks.append(cur.strip())
            cur = (cur[-overlap:] + p + "\n") if overlap else (p + "\n")
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

def process_pdfs(uploaded_files):
    try:
        client.delete_collection("rag_col")
    except Exception:
        pass
    col = client.get_or_create_collection("rag_col")

    ids, docs, metadatas = [], [], []
    for f in uploaded_files:
        reader = pypdf.PdfReader(f)
        for page_num, page in enumerate(reader.pages, start=1):
            for i, c in enumerate(chunk_text(page.extract_text() or "")):
                ids.append(f"{f.name}_p{page_num}_{i}")
                docs.append(c)
                metadatas.append({"source": f.name, "page": page_num})

    col.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embed(docs))
    return col, len(docs)

def suggest_questions(col, n=3):
    """Sinh sẵn vài câu hỏi gợi ý dựa trên nội dung tài liệu vừa index."""
    sample = col.get(limit=6)["documents"]
    context = "\n\n".join(sample)[:4000]
    prompt = f"""Dựa trên đoạn trích tài liệu dưới đây, đề xuất đúng {n} câu hỏi ngắn gọn, hữu ích mà người đọc có thể muốn hỏi.
Chỉ trả về danh sách câu hỏi, mỗi câu một dòng, không đánh số, không giải thích thêm.

Tài liệu:
{context}

Câu hỏi đề xuất:"""
    resp = genai_client.models.generate_content(
        model=LLM_MODEL, contents=prompt, config=types.GenerateContentConfig(temperature=0.7)
    )
    lines = [l.strip("-•*0123456789. ").strip() for l in resp.text.strip().split("\n") if l.strip()]
    return lines[:n]

def rag_stream(question, col, k=4):
    """Truy hồi ngữ cảnh liên quan rồi trả lời dạng streaming, kèm nguồn trích dẫn."""
    res = col.query(
        query_embeddings=embed([question], "RETRIEVAL_QUERY"), n_results=k
    )
    docs = res["documents"][0]
    metadatas = res["metadatas"][0]
    sources = [{"source": m["source"], "page": m["page"], "text": d} for m, d in zip(metadatas, docs)]
    context = "\n\n".join(docs)

    def stream():
        resp = genai_client.models.generate_content_stream(
            model=LLM_MODEL,
            contents=PROMPT.format(context=context, question=question),
            config=types.GenerateContentConfig(temperature=0),
        )
        for chunk in resp:
            if chunk.text:
                yield chunk.text

    return stream(), sources

def render_sources(sources):
    if not sources:
        return
    tags = sorted({f"{s['source']} (trang {s['page']})" for s in sources})
    st.caption("📍 Nguồn: " + " · ".join(tags))
    with st.expander(f"📚 Xem chi tiết {len(sources)} đoạn trích"):
        for s in sources:
            st.markdown(f"**📄 {s['source']} — trang {s['page']}**")
            snippet = s["text"][:300] + ("…" if len(s["text"]) > 300 else "")
            st.caption(snippet)

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
html, body, [class*="css"] { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }

[data-testid="stSidebar"] { border-right: 1px solid #EBEBEA; }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #6B6A66; font-size: 0.9rem; }

[data-testid="stFileUploaderDropzone"] {
    border-radius: 8px;
    border: 1.5px dashed #D9D8D5;
    background: #FBFBFA;
}

[data-testid="stButton"] button {
    border-radius: 6px;
    font-weight: 500;
    border: 1px solid #E4E3E0;
    box-shadow: none;
    transition: background 0.1s ease;
}
[data-testid="stButton"] button:hover {
    background: #F1F0EE;
    border-color: #D9D8D5;
    color: #37352F;
}

[data-testid="stChatMessage"] {
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
    background: #FFFFFF;
    border: 1px solid #EBEBEA;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #F6F4FC;
    border-color: #E6E1F7;
}

[data-testid="stChatInput"] {
    border-radius: 8px;
}
[data-testid="stChatInput"] textarea {
    border-radius: 8px !important;
}

[data-testid="stExpander"] {
    border-radius: 8px;
    border: 1px solid #EBEBEA !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-size: 0.88rem;
    color: #6B6A66;
}

[data-testid="stAlert"] { border-radius: 8px; }

.hero {
    display: flex;
    align-items: baseline;
    gap: 0.65rem;
    padding-bottom: 1rem;
    margin-bottom: 1.3rem;
    border-bottom: 1px solid #EBEBEA;
}
.hero-icon { font-size: 1.7rem; line-height: 1; }
.hero-title { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.01em; color: #37352F; }
.hero-sub { color: #9B9A97; font-size: 0.88rem; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-icon">📄</div>
  <div class="hero-title">PDF RAG Assistant</div>
</div>
<div class="hero-sub" style="margin-top:-1.1rem; margin-bottom:1.3rem;">Hỏi đáp dựa trên tài liệu PDF bạn upload — mọi câu trả lời đều kèm nguồn trích dẫn.</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("📄 Upload tài liệu")
    files = st.file_uploader("Chọn một hoặc nhiều file PDF", type="pdf", accept_multiple_files=True)
    if files and st.button("🔄 Xử lý PDF", use_container_width=True):
        with st.spinner(f"Đang xử lý {len(files)} file..."):
            st.session_state.collection, n = process_pdfs(files)
            st.session_state.pdf_names = [f.name for f in files]
            st.session_state.chat_history = []
            st.session_state.suggested_questions = suggest_questions(st.session_state.collection)
        st.success(f"✅ Đã index {n} đoạn từ {len(files)} tài liệu")

    if st.session_state.pdf_names:
        st.markdown("**Tài liệu đang dùng:**")
        for name in st.session_state.pdf_names:
            st.caption(f"📄 {name}")
    else:
        st.info("📄 Chưa có tài liệu")

    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.chat_history = []

for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("sources"))

if st.session_state.collection is None:
    st.info("🔄 Upload và xử lý ít nhất một PDF trước khi chat.")
    st.chat_input("Nhập câu hỏi...", disabled=True)
else:
    if not st.session_state.chat_history and st.session_state.suggested_questions and not st.session_state.get("pending_question"):
        st.markdown('<div class="hero-sub" style="margin-bottom:0.5rem;">💡 Gợi ý câu hỏi</div>', unsafe_allow_html=True)
        cols = st.columns(len(st.session_state.suggested_questions))
        for c, sq in zip(cols, st.session_state.suggested_questions):
            with c:
                if st.button(sq, use_container_width=True, key=f"sugg_{sq}"):
                    st.session_state.pending_question = sq
                    st.rerun()

    q = st.chat_input("Nhập câu hỏi của bạn...")
    if st.session_state.get("pending_question"):
        q = st.session_state.pop("pending_question")
    if q:
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            stream, sources = rag_stream(q, st.session_state.collection)
            ans = st.write_stream(stream)
            render_sources(sources)
        st.session_state.chat_history.append({"role": "assistant", "content": ans, "sources": sources})