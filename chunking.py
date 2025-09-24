#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone script để chunk tất cả văn bản luật thành chunks
Tương tự như find_law_files.py nhưng cho việc chunking
"""

import os
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

# For AI functionality
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Import từ modules
from modules.chunking import read_docx, chunk_law_document, generate_law_id
from modules.data_loader import save_chunks_to_json

# ====== CẤU HÌNH AGENT ======
GEMINI_MODEL_NAME = "gemini-1.5-flash"  # cố định theo yêu cầu

# ===================== GEMINI EVALUATION AGENT =====================

GEMINI_PROMPT = """Bạn là chuyên gia pháp điển hoá & kiểm thử dữ liệu luật.
Tôi gửi cho bạn:
1) Summary thống kê & cảnh báo từ bộ chunking.
2) Một số "excerpts" (trích đoạn nguyên văn) đại diện.
3) Danh mục chunks (id + metadata + content rút gọn).

Nhiệm vụ:
- PHÁT HIỆN BẤT THƯỜNG (anomalies) trong chunking, ví dụ:
  * Sai thứ tự/chưa "strict" (nhảy cóc Chương/Điều/Khoản/Điểm).
  * Nhận diện nhầm "Khoản" (không phải dạng `1.`) hoặc "Điểm" (không phải `a)`).
  * Không tiêm intro khoản vào điểm khi đã có chuỗi điểm.
  * Thiếu/bỏ sót nội dung so với excerpts.
  * Metadata không khớp: article_no/clause_no/point_letter/exact_citation.
  * Đóng mở chuỗi điểm sai (bắt đầu không phải "a)", chèn nội dung thường vào giữa).
  * Nội dung "Điều" không có khoản nhưng không sinh chunk intro.
- GỢI Ý SỬA: chỉ rõ vị trí (id / exact_citation), mô tả vấn đề, cách khắc phục.
- Nếu KHÔNG thấy vấn đề, xác nhận "ok" và nêu ngắn gọn cơ sở kết luận.

Hãy TRẢ LỜI **CHỈ** ở dạng JSON theo schema:
{
  "status": "ok" | "issues_found",
  "confidence": 0.0-1.0,
  "issues": [
    {
      "id": "chuỗi id chunk hoặc mô tả vị trí",
      "citation": "Điều ... khoản ... điểm ...",
      "severity": "low|medium|high",
      "category": "ordering|regex|metadata|omission|points_chain|format|other",
      "message": "Mô tả ngắn gọn vấn đề",
      "suggestion": "Cách sửa ngắn gọn"
    }
  ],
  "notes": "Ghi chú ngắn (tuỳ chọn)"
}
Trả JSON hợp lệ. Không giải thích ngoài JSON.
"""

def _shorten_text(s: str, max_len: int = 500) -> str:
    if len(s) <= max_len:
        return s
    head = s[: max_len // 2].rstrip()
    tail = s[- max_len // 2 :].lstrip()
    return head + "\n...\n" + tail

def build_review_payload(chunks: list, summary: dict, raw_texts: list, sample_excerpts_chars: int = 2000, max_chunks_sample: int = 50):
    """
    Xây dựng payload cho AI review với sampling thông minh để tránh vượt token limit.

    Args:
        chunks: Danh sách tất cả chunks
        summary: Thống kê tổng quan
        raw_texts: List các raw text từ files mẫu
        sample_excerpts_chars: Tổng ký tự excerpts (chia đều cho các files)
        max_chunks_sample: Số lượng chunks tối đa để sample
    """

    # 1. Sample excerpts từ raw texts (chia đều chars cho từng file)
    excerpts_per_file = sample_excerpts_chars // max(len(raw_texts), 1)
    excerpts_list = []

    for i, raw_text in enumerate(raw_texts[:3]):  # Tối đa 3 files
        if len(raw_text) <= excerpts_per_file:
            excerpts_list.append(f"File {i+1}:\n{raw_text}")
        else:
            # Sample 3 phần: đầu, giữa, cuối
            k = excerpts_per_file // 3
            n = len(raw_text)
            excerpt = raw_text[:k] + "\n...\n" + raw_text[n//2 - k//2 : n//2 + k//2] + "\n...\n" + raw_text[-k:]
            excerpts_list.append(f"File {i+1}:\n{excerpt}")

    combined_excerpts = "\n\n".join(excerpts_list)

    # 2. Sample chunks thông minh (ưu tiên chunks có vấn đề tiềm ẩn)
    def lite(c):
        return {
            "id": c.get("id"),
            "metadata": c.get("metadata"),
            "content_preview": _shorten_text(c.get("content",""), 500)  # Giảm từ 700 xuống 500
        }

    # Sampling strategy: lấy mix của các loại chunks
    sampled_chunks = []
    article_chunks = []
    clause_chunks = []
    point_chunks = []

    for c in chunks:
        metadata = c.get('metadata', {})
        if metadata.get('point_letter'):
            point_chunks.append(c)
        elif metadata.get('clause_no'):
            clause_chunks.append(c)
        elif metadata.get('article_no'):
            article_chunks.append(c)

    # Sample đều từ mỗi loại (tối đa max_chunks_sample)
    samples_per_type = max_chunks_sample // 3

    import random
    sampled_chunks.extend(random.sample(article_chunks, min(samples_per_type, len(article_chunks))))
    sampled_chunks.extend(random.sample(clause_chunks, min(samples_per_type, len(clause_chunks))))
    sampled_chunks.extend(random.sample(point_chunks, min(samples_per_type, len(point_chunks))))

    # Nếu vẫn chưa đủ, thêm random từ tất cả
    remaining = max_chunks_sample - len(sampled_chunks)
    if remaining > 0:
        other_chunks = [c for c in chunks if c not in sampled_chunks]
        sampled_chunks.extend(random.sample(other_chunks, min(remaining, len(other_chunks))))

    # Shuffle để tránh bias
    random.shuffle(sampled_chunks)
    chunks_lite = [lite(c) for c in sampled_chunks]

    return {
        "summary": summary,
        "excerpts": combined_excerpts,
        "chunks_preview": chunks_lite,
        "note": f"Sampled {len(chunks_lite)}/{len(chunks)} chunks from {len(raw_texts)} files for AI review"
    }

def call_gemini_review(payload: dict, api_key: str = None) -> dict:
    """
    Gọi Gemini (model cố định GEMINI_MODEL_NAME) và ép trả JSON.
    API key lấy từ .env (GEMINI_API_KEY).
    """
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Thiếu GEMINI_API_KEY trong môi trường (.env).")

    # import tại chỗ, chỉ khi --AI bật
    import google.generativeai as genai
    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "response_mime_type": "application/json",
    }
    model = genai.GenerativeModel(GEMINI_MODEL_NAME, generation_config=generation_config)
    prompt_parts = [
        {"role": "user", "parts": [{"text": GEMINI_PROMPT}]},
        {"role": "user", "parts": [{"text": json.dumps(payload, ensure_ascii=False)}]},
    ]
    resp = model.generate_content(prompt_parts)
    raw = getattr(resp, "text", None) or (
        resp.candidates and resp.candidates[0].content.parts[0].text
    )
    if not raw:
        raise RuntimeError("Gemini không trả ra nội dung.")
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("JSON không phải object")
        data.setdefault("status", "issues_found")
        data.setdefault("issues", [])
        data.setdefault("confidence", 0.0)
        return data
    except Exception as e:
        return {
            "status": "issues_found",
            "confidence": 0.0,
            "issues": [{
                "id": "PARSER",
                "citation": "",
                "severity": "high",
                "category": "other",
                "message": f"Không parse được JSON từ Gemini: {e}",
                "suggestion": "Chạy lại hoặc giảm excerpts."
            }],
            "notes": raw[:2000]
        }

def load_law_file_paths(json_path="data_files/law_file_paths.json"):
    """Load danh sách file luật từ JSON"""
    if not os.path.exists(json_path):
        print(f"❌ Law file paths JSON not found: {json_path}")
        print("   Please run find_law_files.py first")
        return []

    print(f"📖 Loading law file paths from: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            law_file_paths = json.load(f)
        print(f"✅ Loaded {len(law_file_paths)} law file paths")
        return law_file_paths
    except Exception as e:
        print(f"❌ Error loading {json_path}: {e}")
        return []

def chunk_all_law_documents(law_file_paths):
    """Chunk tất cả văn bản luật từ danh sách file paths"""

    print(f"\n🔍 Starting to chunk {len(law_file_paths)} law documents...")

    all_chunks = []
    successful_chunks = 0
    failed_files = 0
    warnings = []
    articles_count = 0
    article_intro_count = 0
    clauses_count = 0
    points_count = 0

    for i, file_info in enumerate(law_file_paths):
        file_path = file_info['path']
        category = file_info['category']
        file_name = file_info['file_name']

        print(f"\n📄 [{i+1}/{len(law_file_paths)}] Processing: {file_name}")
        print(f"   Category: {category}")
        print(f"   Path: {file_path}")

        if not os.path.exists(file_path):
            print(f"   ❌ File not found: {file_path}")
            failed_files += 1
            continue

        try:
            # Bước 1: Đọc file
            print("   📖 Reading file...")
            raw_text = read_docx(file_path)

            if not raw_text or len(raw_text.strip()) < 100:
                print(f"   ⚠️ File seems empty or too short: {len(raw_text)} characters")
                print("   ⏭️ Skipping file (cannot read content)")
                failed_files += 1
                continue

            # Bước 2: Tạo law_id từ tên file
            law_id = generate_law_id(file_name)
            print(f"   📋 Generated law_id: {law_id}")

            # Bước 3: Chunk document
            print("   🔨 Chunking document...")
            chunks = chunk_law_document(
                text=raw_text,
                law_id=law_id,
                law_no="",
                law_title=file_name
            )

            if not chunks:
                print("   ⚠️ No chunks created from file")
                failed_files += 1
                continue

            # Bước 4: Thêm metadata và chuẩn bị cho save
            print("   🗂️ Preparing chunks...")
            for j, chunk in enumerate(chunks):
                # Thêm thông tin về file gốc vào metadata
                chunk_metadata = chunk.get('metadata', {}).copy()
                chunk_metadata.update({
                    'source_file': file_path,
                    'source_category': category,
                    'source_file_name': file_name,
                    'chunk_index': j
                })

                # Tạo document theo format chunks.json
                all_chunks.append({
                    'id': chunk['id'],
                    'content': chunk['content'],
                    'metadata': chunk_metadata
                })

                # Đếm loại chunk
                metadata = chunk.get('metadata', {})
                if metadata.get('point_letter'):
                    points_count += 1
                elif metadata.get('clause_no'):
                    clauses_count += 1
                elif metadata.get('article_no') and not metadata.get('clause_no'):
                    if 'intro' in chunk['id'].lower():
                        article_intro_count += 1
                    else:
                        articles_count += 1

            print(f"   ✅ Successfully processed {len(chunks)} chunks")
            successful_chunks += len(chunks)

        except Exception as e:
            print(f"   ❌ Error processing file: {e}")
            failed_files += 1
            continue

    print(f"\n📊 Chunking Summary:")
    print(f"   ✅ Successfully chunked: {len(law_file_paths) - failed_files} files")
    print(f"   ❌ Failed to process: {failed_files} files")
    print(f"   📄 Total chunks created: {len(all_chunks)}")

    if all_chunks:
        print(f"   📊 Average chunk length: {sum(len(c['content']) for c in all_chunks) // len(all_chunks):.0f} characters")

        # Thống kê theo category
        category_counts = {}
        for chunk in all_chunks:
            category = chunk.get('metadata', {}).get('source_category', 'Unknown')
            category_counts[category] = category_counts.get(category, 0) + 1

        print("\n📈 Chunks distribution by category:")
        for category, count in category_counts.items():
            print(f"   - {category}: {count} chunks")

    # Tạo summary dictionary
    summary = {
        "chapters_seen": [],  # Không có thông tin chapters trong batch processing
        "articles": articles_count,
        "article_intro": article_intro_count,
        "clauses": clauses_count,
        "points": points_count,
        "citations": [],  # Không có citations trong batch processing
        "warnings": warnings,
        "halted_reason": None,
        "total_chunks": len(all_chunks)
    }

    return all_chunks, summary

def main():
    """Main function để chạy chunking standalone"""

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Chunk luật Việt Nam + (tuỳ chọn) gọi Gemini để thẩm định (--AI).\n"
        "AI review sử dụng sampling thông minh để tránh vượt token limit.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--AI", action="store_true", help="BẬT gọi Gemini (mặc định KHÔNG gọi).")
    parser.add_argument("--sample-excerpts", type=int, default=2000, help="Tổng ký tự excerpts từ TẤT CẢ files gửi Gemini (mặc định 2000, giảm từ 8000).")
    parser.add_argument("--max-chunks-sample", type=int, default=50, help="Số chunks tối đa sample để AI review (mặc định 50, giảm từ 1200).")
    parser.add_argument("--max-files-sample", type=int, default=2, help="Số files tối đa lấy raw text để AI review (mặc định 2, giảm từ 5).")
    parser.add_argument("--strict-ok-only", action="store_true", help="Chỉ ghi chunks nếu Gemini trả 'ok' (chỉ khi --AI).")
    parser.add_argument("--api-key", help="Gemini API key (hoặc set GEMINI_API_KEY env var).")

    args = parser.parse_args()

    print("=" * 80)
    print("🔨 VIETNAMESE LAW DOCUMENT CHUNKING")
    if args.AI:
        print("🤖 WITH AI REVIEW ENABLED")
    print("=" * 80)

    # Load .env if available
    if DOTENV_AVAILABLE:
        load_dotenv()

    # 1. Load danh sách file luật
    law_file_paths = load_law_file_paths()

    if not law_file_paths:
        print("❌ Cannot proceed without law file paths!")
        print("   Run find_law_files.py first")
        return

    # 2. Chunk tất cả documents
    all_chunks, summary = chunk_all_law_documents(law_file_paths)

    # Thu thập raw text cho AI review (nếu bật AI)
    raw_texts = []
    if args.AI:
        print(f"\n📖 Collecting raw texts for AI review (max {args.max_files_sample} files)...")
        files_sampled = 0
        for file_info in law_file_paths:
            if files_sampled >= args.max_files_sample:
                break
            file_path = file_info['path']
            if os.path.exists(file_path):
                try:
                    raw_text = read_docx(file_path)
                    if raw_text and len(raw_text.strip()) > 100:
                        # Giới hạn 5k chars mỗi file thay vì 10k
                        raw_texts.append(raw_text[:5000])
                        files_sampled += 1
                        print(f"   📄 Sampled file {files_sampled}: {file_info['file_name']} ({len(raw_text)} chars)")
                except Exception as e:
                    print(f"   ⚠️ Failed to read {file_info['file_name']}: {e}")
                    continue
        print(f"   ✅ Collected raw text excerpts from {len(raw_texts)} files (total ~{sum(len(t) for t in raw_texts)} chars)")

    if not all_chunks:
        print("❌ No chunks were created!")
        return

    # 3. Save chunks ra JSON
    # Tạo tên file theo format: chunk_Giờphútgiây_ngàythángnăm(2 số cuối)
    now = datetime.now()
    timestamp = now.strftime("%H%M%S_%d%m%y")  # HHMMSS_DDMMYY
    chunks_json_path = f"data/chunk_{timestamp}.json"
    print(f"\n💾 Saving {len(all_chunks)} chunks to {chunks_json_path}...")

    try:
        # Tạo thư mục data nếu chưa có
        Path("data").mkdir(exist_ok=True)

        save_chunks_to_json(all_chunks, chunks_json_path)
        print(f"✅ Successfully saved {len(all_chunks)} chunks to {chunks_json_path}")
        print("   (Similar to hn2014_chunks.json format)")

    except Exception as e:
        print(f"❌ Error saving chunks: {e}")
        return

    # Nếu KHÔNG bật --AI: kết thúc tại đây
    if not args.AI:
        print(f"\n🎉 Chunking completed successfully!")
        print(f"   📄 Total chunks: {len(all_chunks)}")
        print(f"   💾 Output file: {chunks_json_path}")
        print(f"\nNext: Run main_refactored.py to evaluate with chunks from {chunks_json_path}")
        return

    # ===== Khi --AI bật: gọi Gemini để thẩm định =====
    payload = build_review_payload(
        chunks=all_chunks,
        summary=summary,
        raw_texts=raw_texts,
        sample_excerpts_chars=args.sample_excerpts,
        max_chunks_sample=args.max_chunks_sample
    )

    print(f"\n🤖 GỌI GEMINI (gemini-1.5-flash) ĐÁNH GIÁ CHUNKING =====")
    print(f"   📊 Sending: {len(payload['chunks_preview'])} sampled chunks + {len(raw_texts)} file excerpts")
    print(f"   💾 Payload size estimate: ~{len(str(payload)) // 1000}KB")
    try:
        review = call_gemini_review(payload, args.api_key)
    except Exception as e:
        print(f"❌ Lỗi gọi Gemini: {e}", file=sys.stderr)
        if args.strict_ok_only:
            print("❌ --strict-ok-only: Không thể thẩm định, dừng lại.")
            return
        else:
            print(f"⚠️ Không thẩm định được. Đã lưu chunks vào: {chunks_json_path}")
            print(f"\n🎉 Chunking completed (without AI review)!")
            print(f"   📄 Total chunks: {len(all_chunks)}")
            print(f"   💾 Output file: {chunks_json_path}")
            return

    status = review.get("status", "issues_found")
    confidence = review.get("confidence", 0.0)
    issues = review.get("issues", []) or []
    notes = review.get("notes", "")

    print(f"- Trạng thái: {status} | Confidence: {confidence:.2f}")
    if notes:
        print(f"- Ghi chú: {notes[:4000]}")

    if issues:
        print("\n⚠️ BÁO CÁO VẤN ĐỀ TỪ AI:")
        for i, it in enumerate(issues, 1):
            print(f"{i:02d}. [{it.get('severity','?')}] ({it.get('category','other')}) "
                  f"{it.get('citation') or it.get('id') or ''}")
            print(f"    - {it.get('message','(no message)')}")
            if it.get('suggestion'):
                print(f"    → Gợi ý: {it['suggestion']}")
        issues_path = chunks_json_path.replace('.json', '.issues.json')
        with open(issues_path, 'w', encoding='utf-8') as f:
            json.dump(review, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Đã ghi báo cáo vấn đề: {issues_path}")

        if args.strict_ok_only:
            print("❌ --strict-ok-only: Không ghi chunks vì AI chưa xác nhận 'ok'.")
            return

    print(f"\n🎉 Chunking completed successfully!")
    print(f"   📄 Total chunks: {len(all_chunks)}")
    print(f"   💾 Output file: {chunks_json_path}")
    if args.AI and status == "ok":
        print("   ✅ AI đã xác nhận: Chunking OK!")
    print(f"\nNext: Run main_refactored.py to evaluate with chunks from {chunks_json_path}")

if __name__ == "__main__":
    main()
