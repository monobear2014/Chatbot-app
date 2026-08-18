# 📄 PDF RAG Chatbot

Chatbot hỏi đáp tài liệu PDF theo kiến trúc RAG (Retrieval-Augmented Generation), dùng **OpenAI API** cho cả embedding và sinh câu trả lời, **ChromaDB** làm vector store và **Streamlit** cho giao diện. Mọi câu trả lời đều kèm **trích dẫn nguồn** (tên file + số trang) và được **stream** theo từng chữ.

🔗 **Demo:** https://chatbot-app-5khn8fg5ouo7e2gtk9eprh.streamlit.app/

## 🚀 Tính năng

- Upload nhiều file PDF cùng lúc, index toàn bộ trong một collection
- Chia đoạn theo trang, giữ metadata `source` + `page` để trích dẫn chính xác
- **Nhận diện bảng**: bảng được trích thành Markdown và index thành chunk riêng (`kind="table"`), giữ đúng quan hệ hàng–cột; vùng bảng bị loại khỏi text để không index trùng
- Truy hồi semantic search, **điều chỉnh top-k (2–8)** ngay trên UI
- Câu trả lời stream real-time, kèm danh sách nguồn và xem được đoạn trích gốc
- **Tự sinh câu hỏi gợi ý** từ nội dung tài liệu vừa upload
- Xoá lịch sử chat / đặt lại toàn bộ session
- **Đánh giá chất lượng RAG bằng RAGAS** với testset tự sinh từ chính tài liệu

## 🏗️ Kiến trúc

```
        ┌─► bảng ──► Markdown (lặp header nếu bảng dài) ──┐
PDF ──►─┤                                                 ├──► chunk
 pdfplumber                                               │    + metadata
        └─► text (đã loại vùng bảng) ──► chunk 1000/200 ──┘   {source, page, kind}
                            │
                            ▼
              text-embedding-3-small ──► ChromaDB (in-memory)
                                              │
        câu hỏi ──► embedding ──────────────►│ top-k đoạn gần nghĩa nhất
                                              ▼
                        prompt (ngữ cảnh + câu hỏi) ──► gpt-4o-mini
                                              ▼
                              câu trả lời (stream) + trích dẫn nguồn
```

## 📁 Cấu trúc

| File | Vai trò |
|---|---|
| [src/rag_core.py](src/rag_core.py) | Pipeline RAG dùng chung: chunking, trích bảng (`extract_page_tables`), embedding, index, `retrieve`, `rag_stream`, `generate_answer`, `suggest_questions` |
| [src/chatbot_app.py](src/chatbot_app.py) | App Streamlit (UI, upload, chat, citations) |
| [src/evaluate_rag.py](src/evaluate_rag.py) | Script đánh giá bằng RAGAS |
| [data/](data/) | PDF mẫu (`YOLOv10_Tutorials.pdf`) |

App và script eval dùng **cùng một** `rag_core.py`, nên điểm số eval phản ánh đúng pipeline mà người dùng trải nghiệm.

## 🎨 Giao diện

UI dựng theo design system trong [design.md](design.md) — hệ "warm-canvas editorial": canvas cream `#faf9f5` (không dùng trắng thuần), accent coral `#cc785c` dùng dè cho CTA, display serif (Cormorant Garamond thay Copernicus, tracking âm), body Inter, số liệu font JetBrains Mono. Token màu/radius/font khai báo trong [.streamlit/config.toml](.streamlit/config.toml); CSS trong app chỉ lo các component design system có mà Streamlit không có sẵn:

| Component (design.md) | Trong app |
|---|---|
| `top-nav` | thanh trên: mark 4 nhánh + wordmark + badge trạng thái index |
| `hero-band` | headline serif + 4 `feature-card` mô tả pipeline (empty state) |
| `pricing-tier-card` (số dùng serif) | hàng metric: số tài liệu / số đoạn / top-k |
| `badge-pill` · `badge-coral` | chip số trang · chip "N BẢNG" ở phần nguồn |
| `product-mockup-card-dark` | bảng trích từ PDF hiện trên surface tối, số liệu mono |
| `caption-uppercase` | nhãn nhóm sidebar, nhãn vai "BẠN"/"TRỢ LÝ" trong chat |

## ⚙️ Cài đặt

```bash
pip install -r requirements.txt
```

Lấy API key tại [OpenAI Platform](https://platform.openai.com/api-keys) (cần bật billing để tránh giới hạn free tier rất thấp), rồi khai báo theo **một** trong hai cách:

```bash
# 1) biến môi trường / file .env
echo "OPENAI_API_KEY=your_key_here" > .env
```

```toml
# 2) .streamlit/secrets.toml  (dùng cho Streamlit Cloud: Settings → Secrets)
OPENAI_API_KEY = "your_key_here"
```

## ▶️ Chạy app

```bash
streamlit run src/chatbot_app.py
```

Upload PDF ở thanh bên → bấm **Xử lý PDF** → đặt câu hỏi.

## 📊 Đánh giá bằng RAGAS

```bash
python src/evaluate_rag.py --pdf data/YOLOv10_Tutorials.pdf --n-questions 12 --k 4
```

Script sẽ: index PDF → tự sinh testset `(question, ground_truth)` từ các chunk → chạy qua pipeline RAG thật → chấm 4 chỉ số và lưu CSV vào `eval_results/`.

| Chỉ số | Đo điều gì |
|---|---|
| `faithfulness` | Câu trả lời có bám vào ngữ cảnh truy hồi được, không bịa |
| `answer_relevancy` | Câu trả lời có đúng trọng tâm câu hỏi |
| `context_precision` | Đoạn truy hồi có liên quan, ít nhiễu |
| `context_recall` | Ngữ cảnh có chứa đủ thông tin để trả lời |

> ⚠️ **Quota:** free tier của OpenAI giới hạn rất thấp (vài request/ngày cho tài khoản chưa nạp tiền) — **cần bật billing** trước khi chạy eval hoặc dùng app thật sự, nếu không sẽ gặp `429 RESOURCE_EXHAUSTED` ngay cả với vài câu hỏi. Pipeline (`rag_core.py`) đã tự động retry (backoff) khi gặp lỗi 5xx/429 thoáng qua, nhưng không retry được nếu quota bị chặn hẳn.

## ⚠️ Giới hạn hiện tại

- **Ảnh trong PDF chưa được đọc.** Chỉ text layer được index, nên nội dung sơ đồ/biểu đồ/ảnh kết quả bị mất (chỉ còn caption). Hỏi về nội dung một hình, bot sẽ trả lời "không biết" — đúng hành vi, nhưng thông tin vẫn thiếu. Muốn hỗ trợ thì cần render trang thành ảnh và cho GPT-4o (multimodal) mô tả, tốn 1 API call mỗi trang.
- **Bảng dạng ảnh (scan/screenshot) không đọc được** — cần OCR.
- Bảng nằm vắt qua 2 trang được index thành 2 chunk riêng, header không tự lặp sang trang sau.
- Vector store là ChromaDB in-memory: restart app là mất index, phải upload lại.

## 🛠️ Tech stack

Python · Streamlit · OpenAI API (`openai`) · ChromaDB · pdfplumber · RAGAS · LangChain (`langchain-openai`, wrapper cho RAGAS)
