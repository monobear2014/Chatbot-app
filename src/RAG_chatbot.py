import pypdf 
import ollama
import chromadb #lưu trữ và tìm kiếm vectorDB

### Đọc file PDF
reader = pypdf.PdfReader("./data/YOLOv10_Tutorials.pdf")

#Ghép nọi dung các trang thành 1 chuỗi text, mỗi trang cách nhau 1 dòng trống "\n"
full_text = "\n" .join(page.extract_text() or "" for page in reader.pages)

print("Số trang:", len(reader.pages))
print("Tổng số ký tự:",len(full_text))

### fixded_size Chunk
def chunk_text(text,size=1000, overlap=200):
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks ,cur = [],""
    for p in paras :
        if len(cur) +len(p) +1 <=size:
            cur += p +"\n"
        else:
            if cur:
                chunks.append(cur.strip())
            cur = (cur[-overlap:] + p +"\n") if overlap else (p+"\n")
    if cur.strip():
        chunks.append(cur.strip())
    return chunks
chunks = chunk_text(full_text)
print("số chunks:" , len(chunks))
print(chunks[0][:300]) # xem 300 ký tự đầu của chunk

### Embedding và lưu vào vector Database
#hàm tạo embedding từ danh sách text
def embed(texts):
    """Chuyển danh sách chuỗi thành danh sách vector."""
    return ollama.embed(model="bge-m3", input=texts)["embeddings"]

#tạo vectordatabase trong bộ nhớ
client = chromadb.Client()
collection = client.get_or_create_collection("rag")

#Thêm tất cả chunks vào database
collection.add(
    ids=[str(i) for i in range(len(chunks))], #ID duy nhất cho mỗi chunk
    docs = chunks,                            # Nội dung text gốc
    embeddings = embed(chunks),               # Vector tương ứng
)
print("Đã index:", collection.count(),"chunks")

### Retrieve: tìm kiếm đoạn liên quan
def retrieve(query, k =4):
    """Tìm k = 4 đoạn văn bản liên quan nhất với câu hỏi."""
    res = collection.query(
        query_embeddings = embed([query]), #Vector hoá câu hỏi
        n_results = k                      # Số kết quả trả về
    )
    return res["docs"][0]

#Thử tìm kiếm
QUERY ="YOLOv10 dùng để làm gì?"
for doc in retrieve(QUERY):
    print(doc[:200])
    print("-" * 40)

### Hỏi đáp với LLM (RAG)
#Mẫu prompt hướng dẫn LLM cách trả lời
PROMPT = """Bạn là trợ lý hỏi đáp. Dùng các đoạn ngữ cảnh dưới đây để trả lời câu hỏi.
Nếu ngữ cảnh không có thông tin, hãy nói là bạn không biết, đừng bịa.
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

Ngữ cảnh:
{context}

Câu hỏi:{Question}

Trả lời:"""

def rag(questions ,k=4):
    """Hàm RAG chính: Tìm context rồi hỏi LLM."""
    context = "\n\n".join(retrieve(question,k))
    resp = ollama.chat(
        model ="vicuna:7b-v1-q5_1",
        messages = [{"role":"user","content": PROMPT.format(context = context,question=question)}],
        options ={"temparature": 0},#temparature để kiểm soát mức "sáng tạo" của LLM, từ 0 -> 1
    )
    return resp["message"]["content"]

#chạy thử
print(rag("YOLOv10 là gì"))





