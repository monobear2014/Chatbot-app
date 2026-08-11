"""Logic RAG dùng chung cho app Streamlit (chatbot_app.py) và script đánh giá (evaluate_rag.py).

Tách riêng module này để đảm bảo eval đo đúng pipeline (chunking, embedding, prompt)
mà người dùng thực sự trải nghiệm trên app.
"""
import os
import re

import chromadb
import pdfplumber
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
Ngữ cảnh có thể chứa bảng ở dạng Markdown — hãy đọc theo đúng quan hệ hàng/cột khi trả lời.
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


# --- Trích xuất bảng -------------------------------------------------------
# pdfplumber tìm bảng theo đường kẻ, nhưng nhiều PDF vẽ ô bằng hình chữ nhật tô màu
# và chỉ một phần hàng có ô -> một bảng bị vỡ thành nhiều mảnh và mất hàng.
# Cách xử lý: gộp các mảnh thành một vùng, lấy biên cột từ các ô, biên hàng từ vị trí
# dòng chữ, rồi extract lại trên lưới explicit đó.
_FRAG_SETTINGS = {"vertical_strategy": "lines", "horizontal_strategy": "text"}
_REGION_GAP = 40   # khoảng cách dọc tối đa (pt) để gộp 2 mảnh thành cùng một bảng
_ROW_GAP = 6       # khoảng cách dọc tối thiểu (pt) để coi là 2 hàng khác nhau
_MIN_COLS = 2      # bảng 1 cột thường là code block có viền -> bỏ qua
_MIN_ROWS = 2
_CAPTION_RE = re.compile(r"^\s*(?:Bảng|Table)\s*\d+[:.]")


def _page_captions(text):
    """Tìm chú thích bảng ('Bảng 1: ...'), nối dòng kế tiếp nếu chú thích bị ngắt dòng."""
    lines = (text or "").split("\n")
    captions = []
    for i, line in enumerate(lines):
        if not _CAPTION_RE.match(line):
            continue
        cap = line.strip()
        if not cap.endswith(".") and i + 1 < len(lines):
            cap += " " + lines[i + 1].strip()
        captions.append(cap)
    return captions


def _merge_regions(frags):
    """Gộp các mảnh bảng chồng nhau theo chiều ngang và gần nhau theo chiều dọc."""
    regions = []
    for f in sorted(frags, key=lambda f: f.bbox[1]):
        x0, top, x1, bottom = f.bbox
        if regions:
            px0, ptop, px1, pbottom = regions[-1]
            if top - pbottom <= _REGION_GAP and not (x1 < px0 or x0 > px1):
                regions[-1] = (min(px0, x0), ptop, max(px1, x1), max(pbottom, bottom))
                continue
        regions.append((x0, top, x1, bottom))
    return regions


def _column_bounds(page, region):
    """Biên cột suy ra từ các ô (rect) nằm trong vùng bảng."""
    x0, top, x1, bottom = region
    xs = set()
    for r in page.rects:
        if (r["top"] >= top - _REGION_GAP and r["bottom"] <= bottom + _REGION_GAP
                and r["x0"] >= x0 - 3 and r["x1"] <= x1 + 3):
            xs.update((round(r["x0"], 1), round(r["x1"], 1)))
    return sorted(xs)


def _row_bounds(crop):
    """Biên hàng suy ra từ toạ độ dòng chữ (bắt được cả hàng không có đường kẻ)."""
    tops = sorted({round(w["top"]) for w in crop.extract_words()})
    if not tops:
        return []
    bounds = [tops[0] - 2]
    prev = tops[0]
    for t in tops[1:]:
        if t - prev > _ROW_GAP:
            bounds.append((t + prev) / 2)
        prev = t
    bounds.append(tops[-1] + 14)
    return bounds


def table_to_markdown(rows, title=""):
    """Chuyển bảng thành Markdown; hàng đầu làm header."""
    clean = []
    for row in rows:
        cells = [(c or "").replace("\n", " ").replace("|", r"\|").strip() for c in row]
        if any(cells):
            clean.append(cells)
    if len(clean) < _MIN_ROWS:
        return ""
    header, *body = clean
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return (f"{title}\n" if title else "") + "\n".join(lines)


def extract_page_tables(page):
    """Trả về (markdowns, bboxes) của các bảng trên một trang."""
    captions = _page_captions(page.extract_text())
    markdowns, bboxes = [], []
    for region in _merge_regions(page.find_tables(_FRAG_SETTINGS)):
        cols = _column_bounds(page, region)
        if len(cols) - 1 < _MIN_COLS:
            continue
        bbox = (max(cols[0] - 3, 0), max(region[1] - 6, 0),
                min(cols[-1] + 3, page.width), min(region[3] + 6, page.height))
        crop = page.crop(bbox)
        rows = _row_bounds(crop)
        if len(rows) - 1 < _MIN_ROWS:
            continue
        table = crop.extract_table({
            "vertical_strategy": "explicit", "explicit_vertical_lines": cols,
            "horizontal_strategy": "explicit", "explicit_horizontal_lines": rows,
        })
        title = captions[len(markdowns)].strip() if len(markdowns) < len(captions) else ""
        md = table_to_markdown(table or [], title)
        if md:
            markdowns.append(md)
            bboxes.append(bbox)
    return markdowns, bboxes


def chunk_table(md, size=1000):
    """Bảng dài thì cắt theo hàng và lặp lại tiêu đề + header ở mỗi phần."""
    if len(md) <= size:
        return [md]
    lines = md.split("\n")
    head = lines[:3] if lines[0].startswith("|") else lines[:4]
    body, prefix = lines[len(head):], "\n".join(head)
    chunks, cur = [], []
    for row in body:
        if cur and len(prefix) + sum(len(r) + 1 for r in cur) + len(row) > size:
            chunks.append(prefix + "\n" + "\n".join(cur))
            cur = []
        cur.append(row)
    if cur:
        chunks.append(prefix + "\n" + "\n".join(cur))
    return chunks


def _index_pdfs(collection_name, named_sources):
    """named_sources: list[(tên file, đường dẫn hoặc file-like object)]"""
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.get_or_create_collection(collection_name)

    ids, docs, metadatas = [], [], []
    for name, src in named_sources:
        with pdfplumber.open(src) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables, bboxes = extract_page_tables(page)

                # Text của trang: loại bỏ vùng bảng để không index trùng nội dung
                body = page
                for bbox in bboxes:
                    body = body.outside_bbox(bbox)
                for i, c in enumerate(chunk_text(body.extract_text() or "")):
                    ids.append(f"{name}_p{page_num}_t{i}")
                    docs.append(c)
                    metadatas.append({"source": name, "page": page_num, "kind": "text"})

                for ti, md in enumerate(tables):
                    for i, c in enumerate(chunk_table(md)):
                        ids.append(f"{name}_p{page_num}_tbl{ti}_{i}")
                        docs.append(c)
                        metadatas.append({"source": name, "page": page_num, "kind": "table"})

    col.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embed(docs))
    return col, len(docs)


def process_pdf_files(uploaded_files, collection_name="rag_col"):
    """Index các file PDF upload từ Streamlit (file-like object có .name)."""
    return _index_pdfs(collection_name, [(f.name, f) for f in uploaded_files])


def process_pdf_paths(paths, collection_name="rag_eval"):
    """Index các file PDF từ đường dẫn trên đĩa (dùng cho evaluate_rag.py)."""
    return _index_pdfs(collection_name, [(os.path.basename(p), p) for p in paths])


def retrieve(question, col, k=4):
    """Truy hồi context: trả về (docs, metadatas, sources)."""
    res = col.query(query_embeddings=embed([question], "RETRIEVAL_QUERY"), n_results=k)
    docs = res["documents"][0]
    metadatas = res["metadatas"][0]
    sources = [{"source": m["source"], "page": m["page"], "kind": m.get("kind", "text"), "text": d}
               for m, d in zip(metadatas, docs)]
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
