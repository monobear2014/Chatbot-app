"""UI Streamlit cho PDF RAG chatbot.

Giao diện theo design system trong design.md: canvas cream, accent coral, display
serif (Cormorant Garamond thay cho Copernicus), body Inter, mono JetBrains,
bảng số liệu đặt trên surface tối. Token màu/radius/font khai báo trong
.streamlit/config.toml; CSS dưới đây chỉ lo những component design system có mà
Streamlit không có sẵn (topbar, hero, feature card, badge, table card).
"""
import re
from html import escape

import streamlit as st

from rag_core import (
    EMBED_MODEL,
    GEMINI_API_KEY,
    LLM_MODEL,
    process_pdf_files,
    rag_stream,
    suggest_questions,
)

st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄",
                   layout="wide", initial_sidebar_state="expanded")

if not GEMINI_API_KEY:
    st.error("Thiếu GEMINI_API_KEY. Hãy thêm vào .streamlit/secrets.toml (local) hoặc Settings → Secrets (Streamlit Cloud).")
    st.stop()


def default_state():
    """Trạng thái khởi tạo (dict mới mỗi lần gọi để không chia sẻ list giữa các lần reset)."""
    return {
        "collection": None,
        "pdf_names": [],
        "n_chunks": 0,
        "chat_history": [],
        "suggested_questions": [],
        "pending_question": None,
    }


for _k, _v in default_state().items():
    st.session_state.setdefault(_k, _v)


# --- Design system: token + component CSS ----------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --primary: #cc785c;
    --primary-active: #a9583e;
    --ink: #141413;
    --body: #3d3d3a;
    --muted: #6c6a64;
    --muted-soft: #8e8b82;
    --hairline: #e6dfd8;
    --hairline-soft: #ebe6df;
    --canvas: #faf9f5;
    --surface-soft: #f5f0e8;
    --surface-card: #efe9de;
    --surface-cream-strong: #e8e0d2;
    --surface-dark: #181715;
    --surface-dark-soft: #1f1e1b;
    --on-dark: #faf9f5;
    --on-dark-soft: #a09d96;
    --accent-teal: #5db8a6;
    --serif: 'Cormorant Garamond', 'EB Garamond', Garamond, serif;
    --sans: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --mono: 'JetBrains Mono', ui-monospace, monospace;
}

/* Streamlit chrome: bỏ dải gradient mặc định, thu gọn padding trên */
[data-testid="stDecoration"] { display: none; }
.block-container { padding-top: 2.2rem; max-width: 1200px; }

/* typography.display-* — serif 400/500, tracking âm là bắt buộc theo design.
   Cormorant Garamond mặc định dùng chữ số old-style ("10" đọc thành "1o"),
   nên mọi chỗ dùng serif phải bật lining figures. */
h1, h2, h3, .display, .hero-title, .metric .val, .card .step,
.st-key-suggs .stButton button p {
    font-variant-numeric: lining-nums;
    font-feature-settings: "lnum" 1, "onum" 0;
}
h1, h2, h3, .display { font-family: var(--serif); font-weight: 500; color: var(--ink); }

/* --- top-nav ---------------------------------------------------------- */
.topbar {
    display: flex; align-items: center; gap: 10px;
    padding-bottom: 14px; margin-bottom: 28px;
    border-bottom: 1px solid var(--hairline);
}
.wordmark { font-family: var(--sans); font-weight: 500; font-size: 15px; color: var(--ink); letter-spacing: 0; }
.wordmark span { color: var(--muted); font-weight: 400; }
.topbar .spacer { flex: 1; }

/* badge-pill / badge-coral */
.badge {
    font-family: var(--sans); font-size: 13px; font-weight: 500;
    background: var(--surface-card); color: var(--ink);
    border-radius: 9999px; padding: 4px 12px; white-space: nowrap;
}
.badge-coral {
    background: var(--primary); color: #fff;
    font-size: 12px; letter-spacing: 1.5px; text-transform: uppercase;
}
.badge-dot::before {
    content: ''; display: inline-block; width: 6px; height: 6px;
    border-radius: 50%; background: var(--accent-teal); margin-right: 7px;
    vertical-align: middle;
}

/* --- hero-band -------------------------------------------------------- */
.hero-title {
    font-family: var(--serif); font-weight: 500; font-size: 44px;
    line-height: 1.1; letter-spacing: -0.02em; color: var(--ink); margin: 0;
}
.hero-sub {
    font-family: var(--sans); font-size: 16px; line-height: 1.55;
    color: var(--muted); margin: 10px 0 0; max-width: 60ch;
}

/* caption-uppercase — nhãn nhóm trong sidebar và trên các dải nội dung */
.eyebrow {
    font-family: var(--sans); font-size: 12px; font-weight: 500;
    letter-spacing: 1.5px; text-transform: uppercase; color: var(--muted-soft);
    margin: 0 0 10px;
}

/* --- feature-card (empty state) --------------------------------------- */
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(215px, 1fr)); gap: 16px; margin-top: 32px; }
.card {
    background: var(--surface-card); border-radius: 12px; padding: 24px;
    font-family: var(--sans); font-size: 14px; line-height: 1.55; color: var(--body);
}
.card .step { font-family: var(--serif); font-size: 26px; color: var(--primary); line-height: 1; }
.card .name { font-family: var(--sans); font-size: 16px; font-weight: 500; color: var(--ink); margin: 12px 0 6px; }

/* --- metric row: số liệu dùng serif display theo tinh thần pricing card -- */
.metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 26px; }
.metric { background: var(--canvas); border: 1px solid var(--hairline); border-radius: 12px; padding: 18px 20px; }
.metric .val { font-family: var(--serif); font-weight: 500; font-size: 30px; line-height: 1; letter-spacing: -0.02em; color: var(--ink); }
.metric .lbl { font-family: var(--sans); font-size: 12px; font-weight: 500; letter-spacing: 1.5px; text-transform: uppercase; color: var(--muted-soft); margin-top: 8px; }

/* --- chat: user trên cream card, assistant trên canvas + hairline -------
   Design system không có avatar; phân biệt vai bằng khối màu + nhãn eyebrow,
   nên avatar emoji mặc định của Streamlit được ẩn đi. */
[data-testid="stChatMessage"] {
    flex-direction: column; align-items: stretch; gap: 10px;
    border-radius: 12px; padding: 18px 22px; margin-bottom: 10px;
    background: var(--canvas); border: 1px solid var(--hairline-soft);
}
/* xếp dọc rồi thì khối nội dung phải chiếm hết chiều ngang, không canh giữa */
[data-testid="stChatMessage"] > [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"] > div { width: 100%; text-align: left; }
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] { display: none; }
[data-testid="stChatMessage"]::before {
    font-family: var(--sans); font-size: 12px; font-weight: 500;
    letter-spacing: 1.5px; text-transform: uppercase; color: var(--muted-soft);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: var(--surface-card); border-color: transparent;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])::before { content: 'Bạn'; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])::before { content: 'Trợ lý'; }
[data-testid="stChatMessage"] p { font-family: var(--sans); color: var(--body); }

/* --- nguồn trích dẫn -------------------------------------------------- */
.cites { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-top: 6px; }
.cites .lbl { font-family: var(--sans); font-size: 12px; font-weight: 500; letter-spacing: 1.5px; text-transform: uppercase; color: var(--muted-soft); margin-right: 4px; }

/* --- product-mockup-card-dark: bảng trích từ tài liệu ------------------ */
.tbl-card { background: var(--surface-dark); border-radius: 12px; padding: 22px; margin: 6px 0 14px; }
.tbl-title { font-family: var(--sans); font-size: 13px; font-weight: 500; color: var(--on-dark-soft); line-height: 1.45; margin-bottom: 14px; }
.tbl-scroll { overflow-x: auto; }
/* Streamlit kẻ viền mọi ô của table -> reset trước rồi chỉ giữ đường ngang */
.tbl-card table, .tbl-card th, .tbl-card td { border: none !important; background: transparent !important; }
.tbl-card table { border-collapse: collapse; width: 100%; font-family: var(--mono); font-size: 13px; }
.tbl-card th {
    font-family: var(--sans); font-size: 12px; font-weight: 500; letter-spacing: 1px;
    text-transform: uppercase; color: var(--on-dark-soft) !important; text-align: left;
    padding: 0 22px 10px 0; border-bottom: 1px solid #34322d !important; white-space: nowrap;
}
.tbl-card td {
    color: var(--on-dark) !important; padding: 9px 22px 9px 0;
    border-bottom: 1px solid var(--surface-dark-soft) !important; white-space: nowrap;
}
.tbl-card tr:last-child td { border-bottom: none !important; }

/* --- đoạn trích text -------------------------------------------------- */
.snippet {
    background: var(--surface-soft); border-radius: 8px; padding: 14px 16px; margin-bottom: 12px;
    font-family: var(--sans); font-size: 13px; line-height: 1.6; color: var(--body);
}
.snippet .src { font-size: 12px; font-weight: 500; letter-spacing: 1px; text-transform: uppercase; color: var(--muted-soft); display: block; margin-bottom: 7px; }
/* code mặc định của Streamlit màu xanh lá — lệch khỏi bảng màu cream/coral */
.snippet code { font-family: var(--mono); font-size: 12.5px; color: var(--ink) !important; background: transparent !important; padding: 0; }

/* --- widget: bám radius/hairline của design, không thêm hover ---------- */
[data-testid="stFileUploaderDropzone"] { background: var(--canvas); border: 1.5px dashed var(--hairline); border-radius: 8px; }
[data-testid="stExpander"] { border: 1px solid var(--hairline) !important; border-radius: 12px; overflow: hidden; background: var(--canvas); }
[data-testid="stExpander"] summary { font-family: var(--sans); font-size: 13px; font-weight: 500; color: var(--muted); }
[data-testid="stChatInput"] textarea { font-family: var(--sans); }
[data-testid="stSidebar"] hr { border-color: var(--hairline); margin: 18px 0; }
.stButton button { font-family: var(--sans); font-weight: 500; font-size: 14px; }
.stButton button:active { background: var(--surface-cream-strong); }
.stButton button[kind="primary"]:active { background: var(--primary-active); }
/* gợi ý câu hỏi: canh lề trái như danh sách bài đọc, không phải nút CTA */
.st-key-suggs .stButton button { justify-content: flex-start; text-align: left; padding: 12px 16px; color: var(--ink); }
.st-key-suggs .stButton button p { font-family: var(--serif) !important; font-size: 17px !important; font-weight: 500; letter-spacing: -0.01em; }
</style>
""", unsafe_allow_html=True)

# Dấu 4 nhánh (radial spike) dùng làm mark cạnh wordmark, theo design system
SPIKE = """<svg width="17" height="17" viewBox="0 0 24 24" fill="#141413">
<path d="M12 1.5 13.5 9.6 12 12 10.5 9.6Z"/><path d="M12 22.5 10.5 14.4 12 12 13.5 14.4Z"/>
<path d="M1.5 12 9.6 10.5 12 12 9.6 13.5Z"/><path d="M22.5 12 14.4 13.5 12 12 14.4 10.5Z"/>
<path d="M4.6 4.6 10.9 9.5 12 12 9.5 10.9Z"/><path d="M19.4 19.4 13.1 14.5 12 12 14.5 13.1Z"/>
<path d="M19.4 4.6 14.5 10.9 12 12 13.1 9.5Z"/><path d="M4.6 19.4 9.5 13.1 12 12 10.9 14.5Z"/>
</svg>"""

_SEP_CHARS = set("-: ")


def render_table_card(md):
    """Bảng Markdown do rag_core sinh ra -> card tối, số liệu font mono."""
    lines = [l for l in md.strip().split("\n") if l.strip()]
    title = "" if lines[0].startswith("|") else lines[0]
    rows = []
    for line in (l for l in lines if l.startswith("|")):
        cells = [c.strip().replace(r"\|", "|") for c in re.split(r"(?<!\\)\|", line)[1:-1]]
        if cells and all(c and set(c) <= _SEP_CHARS for c in cells):
            continue                      # bỏ hàng phân cách '| --- |'
        rows.append(cells)
    if not rows:
        st.markdown(md)
        return
    head, *body = rows
    html = ['<div class="tbl-card">']
    if title:
        html.append(f'<div class="tbl-title">{escape(title)}</div>')
    html.append('<div class="tbl-scroll"><table><thead><tr>')
    html += [f"<th>{escape(c)}</th>" for c in head]
    html.append("</tr></thead><tbody>")
    for row in body:
        html.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in row) + "</tr>")
    html.append("</tbody></table></div></div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_sources(sources):
    if not sources:
        return
    pages = sorted({s["page"] for s in sources})
    n_tbl = sum(1 for s in sources if s.get("kind") == "table")
    chips = [f'<span class="badge">trang {p}</span>' for p in pages]
    if n_tbl:
        chips.append(f'<span class="badge badge-coral">{n_tbl} bảng</span>')
    st.markdown('<div class="cites"><span class="lbl">Nguồn</span>' + "".join(chips) + "</div>",
                unsafe_allow_html=True)

    with st.expander(f"Xem {len(sources)} đoạn trích"):
        for s in sources:
            if s.get("kind") == "table":
                render_table_card(s["text"])
            else:
                snippet = s["text"][:320] + ("…" if len(s["text"]) > 320 else "")
                st.markdown(f'<div class="snippet"><span class="src">{escape(s["source"])} '
                            f'· trang {s["page"]}</span>{escape(snippet)}</div>',
                            unsafe_allow_html=True)


# --- top-nav ---------------------------------------------------------------
ready = st.session_state.collection is not None
status = ('<span class="badge badge-dot">Đã index '
          f'{st.session_state.n_chunks} đoạn</span>' if ready else
          '<span class="badge">Chưa có tài liệu</span>')
st.markdown(f'<div class="topbar">{SPIKE}'
            f'<div class="wordmark">PDF RAG <span>· Assistant</span></div>'
            f'<div class="spacer"></div>{status}</div>', unsafe_allow_html=True)

# --- sidebar ---------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="eyebrow">Tài liệu</div>', unsafe_allow_html=True)
    files = st.file_uploader("Chọn một hoặc nhiều file PDF", type="pdf",
                             accept_multiple_files=True, label_visibility="collapsed")
    if files and st.button("Xử lý PDF", use_container_width=True, type="primary"):
        with st.spinner(f"Đang đọc, tách bảng và tạo embedding cho {len(files)} file..."):
            st.session_state.collection, st.session_state.n_chunks = process_pdf_files(files)
            st.session_state.pdf_names = [f.name for f in files]
            st.session_state.chat_history = []
            st.session_state.suggested_questions = suggest_questions(st.session_state.collection)
        st.rerun()

    # Tên file không lặp lại ở đây: widget uploader phía trên đã liệt kê sẵn.

    st.divider()
    st.markdown('<div class="eyebrow">Truy hồi</div>', unsafe_allow_html=True)
    top_k = st.slider("Số đoạn ngữ cảnh (top-k)", 2, 8, 4,
                      help="Số đoạn liên quan nhất được đưa vào prompt cho LLM.")

    col_a, col_b = st.columns(2)
    if col_a.button("Xoá chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    if col_b.button("Đặt lại", use_container_width=True,
                    help="Xoá tài liệu đã index và toàn bộ lịch sử."):
        st.session_state.update(default_state())
        st.rerun()

    st.divider()
    st.markdown('<div class="eyebrow">Mô hình</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="snippet" style="padding:12px 14px;">'
                f'<span class="src">LLM</span><code>{LLM_MODEL}</code></div>'
                f'<div class="snippet" style="padding:12px 14px;">'
                f'<span class="src">Embedding</span><code>{EMBED_MODEL}</code></div>'
                f'<div class="snippet" style="padding:12px 14px;">'
                f'<span class="src">Vector store</span>ChromaDB (in-memory)</div>',
                unsafe_allow_html=True)

# --- hero / empty state ----------------------------------------------------
if not ready:
    st.markdown(
        '<h1 class="hero-title">Hỏi tài liệu của bạn,<br>nhận câu trả lời có nguồn.</h1>'
        '<p class="hero-sub">Upload PDF ở thanh bên rồi bấm <b>Xử lý PDF</b>. Mọi câu trả lời '
        'đều kèm số trang, và bảng số liệu được đọc theo đúng hàng–cột.</p>'
        '<div class="cards">'
        '<div class="card"><div class="step">1</div><div class="name">Ingest</div>'
        'Đọc PDF bằng pdfplumber, tách riêng bảng, chia text thành đoạn 1000 ký tự (overlap 200).</div>'
        '<div class="card"><div class="step">2</div><div class="name">Index</div>'
        'Tạo embedding cho từng đoạn, lưu vào ChromaDB kèm tên file và số trang.</div>'
        '<div class="card"><div class="step">3</div><div class="name">Retrieve</div>'
        'Tìm top-k đoạn gần nghĩa nhất với câu hỏi — bảng và text cạnh tranh công bằng.</div>'
        '<div class="card"><div class="step">4</div><div class="name">Generate</div>'
        'LLM trả lời dựa trên ngữ cảnh, stream từng chữ, kèm trích dẫn nguồn.</div>'
        '</div>', unsafe_allow_html=True)
    st.chat_input("Nhập câu hỏi...", disabled=True)
    st.stop()

# --- trạng thái đã index ---------------------------------------------------
st.markdown(
    f'<div class="metrics">'
    f'<div class="metric"><div class="val">{len(st.session_state.pdf_names)}</div>'
    f'<div class="lbl">Tài liệu</div></div>'
    f'<div class="metric"><div class="val">{st.session_state.n_chunks}</div>'
    f'<div class="lbl">Đoạn đã index</div></div>'
    f'<div class="metric"><div class="val">{top_k}</div>'
    f'<div class="lbl">Ngữ cảnh mỗi câu</div></div>'
    f'</div>', unsafe_allow_html=True)

# chat_input luôn được ghim ở đáy trang, nên đọc nó trước để biết có câu hỏi mới
# hay không — nhờ vậy khối gợi ý ẩn ngay trong lần chạy này thay vì lần sau.
q = st.chat_input("Nhập câu hỏi của bạn...")
if st.session_state.pending_question:
    q = st.session_state.pending_question
    st.session_state.pending_question = None

if not st.session_state.chat_history and st.session_state.suggested_questions and not q:
    st.markdown('<div class="eyebrow">Gợi ý từ tài liệu của bạn</div>', unsafe_allow_html=True)
    with st.container(key="suggs"):
        for i, sq in enumerate(st.session_state.suggested_questions):
            if st.button(sq, use_container_width=True, key=f"sugg_{i}"):
                st.session_state.pending_question = sq
                st.rerun()

for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("sources"))

if q:
    st.session_state.chat_history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        with st.spinner(f"Đang truy hồi {top_k} đoạn liên quan..."):
            stream, sources = rag_stream(q, st.session_state.collection, k=top_k)
        ans = st.write_stream(stream)
        render_sources(sources)
    st.session_state.chat_history.append({"role": "assistant", "content": ans, "sources": sources})
