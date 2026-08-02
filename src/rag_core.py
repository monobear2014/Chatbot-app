"""Logic RAG dùng chung cho app Streamlit (chatbot_app.py) và script đánh giá (evaluate_rag.py).

Tách riêng module này để đảm bảo eval đo đúng pipeline (chunking, embedding, prompt)
mà người dùng thực sự trải nghiệm trên app.
"""
import os

import chromadb
import pypdf
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def get_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


GEMINI_API_KEY = get_api_key()
genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
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


def embed(texts, task_type="RETRIEVAL_DOCUMENT"):
    resp = genai_client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [e.values for e in resp.embeddings]


def chunk_text(text, size=1000, overlap=200):
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        # Nếu một đoạn dài hơn size, cắt nhỏ đoạn đó (vẫn giữ overlap)
        while len(p) > size:
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


def _index_readers(collection_name, named_readers):
    """named_readers: list[(name, pypdf.PdfReader)]"""
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.get_or_create_collection(collection_name)

    ids, docs, metadatas = [], [], []
    for name, reader in named_readers:
        for page_num, page in enumerate(reader.pages, start=1):
            for i, c in enumerate(chunk_text(page.extract_text() or "")):
                ids.append(f"{name}_p{page_num}_{i}")
                docs.append(c)
                metadatas.append({"source": name, "page": page_num})

    col.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embed(docs))
    return col, len(docs)


def process_pdf_files(uploaded_files, collection_name="rag_col"):
    """Index các file PDF upload từ Streamlit (file-like object có .name)."""
    named_readers = [(f.name, pypdf.PdfReader(f)) for f in uploaded_files]
    return _index_readers(collection_name, named_readers)


def process_pdf_paths(paths, collection_name="rag_eval"):
    """Index các file PDF từ đường dẫn trên đĩa (dùng cho evaluate_rag.py)."""
    named_readers = [(os.path.basename(p), pypdf.PdfReader(p)) for p in paths]
    return _index_readers(collection_name, named_readers)


def retrieve(question, col, k=4):
    """Truy hồi context: trả về (docs, metadatas, sources)."""
    res = col.query(query_embeddings=embed([question], "RETRIEVAL_QUERY"), n_results=k)
    docs = res["documents"][0]
    metadatas = res["metadatas"][0]
    sources = [{"source": m["source"], "page": m["page"], "text": d} for m, d in zip(metadatas, docs)]
    return docs, metadatas, sources


def generate_answer(question, context):
    """Sinh câu trả lời (non-streaming) — dùng cho evaluation."""
    resp = genai_client.models.generate_content(
        model=LLM_MODEL,
        contents=PROMPT.format(context=context, question=question),
        config=types.GenerateContentConfig(temperature=0),
    )
    return resp.text


def rag_stream(question, col, k=4):
    """Truy hồi ngữ cảnh liên quan rồi trả lời dạng streaming, kèm nguồn trích dẫn."""
    docs, _, sources = retrieve(question, col, k)
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
