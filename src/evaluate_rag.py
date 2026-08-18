"""Đánh giá pipeline RAG bằng RAGAS.

Quy trình:
1. Index PDF (dùng đúng pipeline của app: rag_core.py).
2. Tự sinh bộ câu hỏi + đáp án chuẩn (ground truth) từ các đoạn văn bản trong PDF, bằng OpenAI.
3. Chạy từng câu hỏi qua pipeline RAG thật (retrieve + generate) để lấy answer + retrieved_contexts.
4. Chấm điểm bằng RAGAS: faithfulness, answer_relevancy, context_precision, context_recall.
5. In bảng kết quả + lưu CSV.

Chạy: python src/evaluate_rag.py [--pdf data/YOLOv10_Tutorials.pdf] [--n-questions 12] [--k 4]
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import rag_core

# Billing đã bật -> quota cao hơn nhiều so với free tier, chỉ cần giãn nhẹ giữa các
# lệnh gọi để tránh burst, không cần chờ theo giới hạn 5 request/phút của free tier nữa.
RATE_LIMIT_SLEEP = 1

QA_GEN_PROMPT = """Dựa CHỈ vào đoạn văn bản dưới đây, hãy đặt 1 câu hỏi và đưa ra câu trả lời chuẩn cho câu hỏi đó.
Câu hỏi phải hỏi về một thông tin cụ thể có trong đoạn văn bản (không hỏi chung chung).
Câu trả lời phải ngắn gọn, chính xác, chỉ dựa vào đoạn văn bản, không bịa thêm.

Đoạn văn bản:
{chunk}

Trả về đúng định dạng JSON: {{"question": "...", "answer": "..."}}"""


@rag_core.retry_on_overload
def _generate_qa(chunk):
    return rag_core.openai_client.chat.completions.create(
        model=rag_core.LLM_MODEL,
        messages=[{"role": "user", "content": QA_GEN_PROMPT.format(chunk=chunk)}],
        temperature=0.3,
        response_format={"type": "json_object"},
    )


def generate_testset(col, n_questions):
    """Sinh n_questions cặp (question, ground_truth) từ các chunk đã index, trải đều trong tài liệu."""
    all_docs = col.get()["documents"]
    if not all_docs:
        raise ValueError("Collection rỗng, không thể sinh testset.")

    stride = max(1, len(all_docs) // n_questions)
    sampled = all_docs[::stride][:n_questions]

    testset = []
    for chunk in sampled:
        resp = _generate_qa(chunk)
        try:
            qa = json.loads(resp.choices[0].message.content)
            if qa.get("question") and qa.get("answer"):
                testset.append({"question": qa["question"].strip(), "ground_truth": qa["answer"].strip()})
        except (json.JSONDecodeError, AttributeError):
            continue
        time.sleep(RATE_LIMIT_SLEEP)
    return testset


def run_pipeline(testset, col, k):
    """Chạy từng câu hỏi qua pipeline RAG thật để lấy answer + retrieved_contexts."""
    samples = []
    for item in testset:
        docs, _, _ = rag_core.retrieve(item["question"], col, k)
        answer = rag_core.generate_answer(item["question"], "\n\n".join(docs))
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=answer,
                retrieved_contexts=docs,
                reference=item["ground_truth"],
            )
        )
        time.sleep(RATE_LIMIT_SLEEP)
    return samples


def main():
    parser = argparse.ArgumentParser(description="Đánh giá RAG pipeline bằng RAGAS")
    parser.add_argument("--pdf", nargs="+", default=["data/YOLOv10_Tutorials.pdf"], help="Đường dẫn file PDF")
    parser.add_argument("--n-questions", type=int, default=12, help="Số câu hỏi trong testset")
    parser.add_argument("--k", type=int, default=4, help="Số chunk retrieve mỗi câu hỏi")
    parser.add_argument("--out", default="eval_results", help="Thư mục lưu kết quả")
    args = parser.parse_args()

    if not rag_core.OPENAI_API_KEY:
        print("Thiếu OPENAI_API_KEY (đặt trong .env).", file=sys.stderr)
        sys.exit(1)

    print(f"[1/4] Index {len(args.pdf)} PDF...")
    col, n_chunks = rag_core.process_pdf_paths(args.pdf)
    print(f"      Đã index {n_chunks} chunks.")

    print(f"[2/4] Sinh testset ({args.n_questions} câu hỏi) từ nội dung tài liệu...")
    testset = generate_testset(col, args.n_questions)
    print(f"      Sinh được {len(testset)} câu hỏi.")
    if not testset:
        print("Không sinh được câu hỏi nào, dừng lại.", file=sys.stderr)
        sys.exit(1)

    print("[3/4] Chạy pipeline RAG thật cho từng câu hỏi (retrieve + generate)...")
    samples = run_pipeline(testset, col, args.k)

    print("[4/4] Chấm điểm bằng RAGAS...")
    api_key = rag_core.OPENAI_API_KEY
    # Billing đã bật -> quota per-minute cao hơn nhiều free tier, chỉ cần giữ tốc độ vừa
    # phải để tránh burst quá lớn; retry ở RunConfig lo phần dự phòng lỗi mạng/quá tải.
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=5,
        check_every_n_seconds=0.5,
        max_bucket_size=10,
    )
    eval_llm = LangchainLLMWrapper(
        ChatOpenAI(model=rag_core.LLM_MODEL, api_key=api_key, temperature=0, rate_limiter=rate_limiter)
    )
    eval_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=rag_core.EMBED_MODEL, api_key=api_key)
    )
    metrics = [
        Faithfulness(llm=eval_llm),
        ResponseRelevancy(llm=eval_llm, embeddings=eval_embeddings),
        LLMContextPrecisionWithReference(llm=eval_llm),
        LLMContextRecall(llm=eval_llm),
    ]

    # max_workers>1: chạy song song vì quota per-minute giờ đủ cao, rate_limiter ở trên
    # vẫn là hàng rào chính ngăn burst quá lớn.
    run_config = RunConfig(max_workers=4, max_retries=5, max_wait=60, timeout=180)

    dataset = EvaluationDataset(samples=samples)
    result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)
    df = result.to_pandas()

    os.makedirs(args.out, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.out, f"ragas_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False)

    print("\n=== Điểm trung bình ===")
    score_cols = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall"]
    for col_name in score_cols:
        if col_name in df.columns:
            print(f"{col_name:45s}: {df[col_name].mean():.3f}")

    print(f"\nChi tiết từng câu đã lưu tại: {csv_path}")


if __name__ == "__main__":
    main()
