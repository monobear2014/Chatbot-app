import streamlit as st

from rag_core import GEMINI_API_KEY, process_pdf_files, rag_stream, suggest_questions

if not GEMINI_API_KEY:
    st.error("Thiếu GEMINI_API_KEY. Hãy thêm vào .streamlit/secrets.toml (local) hoặc Settings → Secrets (Streamlit Cloud).")
    st.stop()

for k, v in {"collection": None, "pdf_names": [], "chat_history": [], "suggested_questions": []}.items():
    st.session_state.setdefault(k, v)

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
            st.session_state.collection, n = process_pdf_files(files)
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