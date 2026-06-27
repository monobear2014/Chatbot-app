import streamlit as st
import pypdf
import chromadb
import ollama

client = chromadb.Client()
LLM_MODEL = "vicuna:7b-v1.5-q5_1"
EMBED_MODEL = "bge-m3"
PROMPT = """Bạn là trợ lý hỏi đáp. Dùng các đoạn ngữ cảnh dưới đây để trả lời câu hỏi.
Nếu ngữ cảnh không có thông tin, hãy nói là bạn không biết, đừng bịa.
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""

for k, v in {"collection": None, "pdf_name": "", "chat_history": []}.items():
    st.session_state.setdefault(k, v)

def embed(texts):
    return ollama.embed(
        model=EMBED_MODEL,
        input=texts,
    )["embeddings"]

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

def read_file(file):
    reader = pypdf.PdfReader(file)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return full_text

def process_pdf(uploaded_file):
    text = read_file(uploaded_file)
    chunks = chunk_text(text)

    col = client.get_or_create_collection("rag_col")
    col.add(
        ids=[str(i) for i in range(len(chunks))], documents=chunks, embeddings=embed(chunks)
    )
    return col, len(chunks)

def rag(question, col, k=2):
    res = col.query(
        query_embeddings=embed([question]), n_results=k
    )
    context = "\n".join(res["documents"][0])
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=[{
            "role":"user",
            "content": PROMPT.format(context=context, question=question),
            }],
        options={"temprature":0},
    )
    return resp.message.content

st.set_page_config(page_title="PDF RAG Chatbot", layout="wide", initial_sidebar_state="expanded")
st.title("PDF RAG Assistant")

with st.sidebar:
    st.subheader("📄 Upload tài liệu")
    f = st.file_uploader("Chọn file PDF", type="pdf")
    if f and st.button("🔄 Xử lý PDF", use_container_width=True):
        with st.spinner("Đang xử lý..."):
            st.session_state.collection, n = process_pdf(f)
            st.session_state.pdf_name = f.name
            st.session_state.chat_history = []
        st.success(f"✅ {n} chunks")
    st.info(f"📄 {st.session_state.pdf_name}" if st.session_state.pdf_name else "📄 Chưa có tài liệu")
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.chat_history = []

for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if st.session_state.collection is None: # DB
    st.info("🔄 Upload và xử lý PDF trước khi chat.")
    st.chat_input("Nhập câu hỏi...", disabled=True)
else:
    q = st.chat_input("Nhập câu hỏi của bạn...")
    if q:
        st.session_state.chat_history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                ans = rag(q, st.session_state.collection)
                st.write(ans)
        st.session_state.chat_history.append({"role": "assistant", "content": ans})