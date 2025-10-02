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
from typing import List, Dict, Any

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
        print(f"ERROR: Law file paths JSON not found: {json_path}")
        print("   Please run find_law_files.py first")
        return []

    print(f"Loading law file paths from: {json_path}")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            law_file_paths = json.load(f)
        print(f"Loaded {len(law_file_paths)} law file paths")
        return law_file_paths
    except Exception as e:
        print(f"ERROR loading {json_path}: {e}")
        return []

def validate_chunks(chunks, verbose=False):
    """Validate chunks sau khi tạo"""
    if verbose:
        print(f"\nValidating {len(chunks)} chunks...")

    validation_results = {
        "total_chunks": len(chunks),
        "valid_chunks": 0,
        "invalid_chunks": 0,
        "issues": []
    }

    required_fields = ["id", "content", "metadata"]
    required_metadata = ["law_id", "article_no", "exact_citation"]

    for i, chunk in enumerate(chunks):
        is_valid = True
        issues = []

        # Check required fields
        for field in required_fields:
            if field not in chunk:
                issues.append(f"Missing field: {field}")
                is_valid = False

        # Check metadata
        if "metadata" in chunk:
            metadata = chunk["metadata"]
            for field in required_metadata:
                if field not in metadata:
                    issues.append(f"Missing metadata field: {field}")
                    is_valid = False

            # Check content length
            if len(chunk.get("content", "")) < 50:
                issues.append("Content too short (< 50 chars)")
                is_valid = False

            # Check ID format
            chunk_id = chunk.get("id", "")
            if not chunk_id or len(chunk_id.split("-")) < 2:
                issues.append("Invalid ID format")
                is_valid = False

        if is_valid:
            validation_results["valid_chunks"] += 1
        else:
            validation_results["invalid_chunks"] += 1
            validation_results["issues"].append({
                "chunk_index": i,
                "chunk_id": chunk.get("id", "unknown"),
                "issues": issues
            })

            if verbose and issues:
                print(f"  WARNING: Chunk {i} ({chunk.get('id', 'unknown')}): {', '.join(issues)}")

    if verbose:
        print(f"Validation complete: {validation_results['valid_chunks']} valid, {validation_results['invalid_chunks']} invalid")

    return validation_results

def load_law_file_paths_by_category(category_folder):
    """Load danh sách file luật từ category-specific JSON"""
    folder_mapping = {
        "BDS": "BDS",
        "DN": "DN",
        "TM": "TM",
        "QDS": "QDS"
    }

    folder_name = folder_mapping.get(category_folder.upper(), category_folder.upper())
    json_path = f"data_files/{folder_name}/{folder_name.lower()}_file_paths.json"

    return load_law_file_paths(json_path)

def chunk_all_law_documents(law_file_paths, law_no="", issued_date="", effective_date="", signer=""):
    """Chunk tất cả văn bản luật từ danh sách file paths"""

    print(f"\nStarting to chunk {len(law_file_paths)} law documents...")

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

        # Sanitize strings for console output
        safe_filename = file_name.encode('ascii', 'ignore').decode('ascii') or file_name
        safe_category = str(category).encode('ascii', 'ignore').decode('ascii') or str(category)
        safe_path = str(file_path).encode('ascii', 'ignore').decode('ascii') or str(file_path)
        print(f"\n[{i+1}/{len(law_file_paths)}] Processing: {safe_filename}")
        print(f"   Category: {safe_category}")
        print(f"   Path: {safe_path}")

        if not os.path.exists(file_path):
            print(f"   ERROR: File not found: {file_path}")
            failed_files += 1
            continue

        try:
            # Bước 1: Đọc file
            print("   Reading file...")
            raw_text = read_docx(file_path)

            if not raw_text or len(raw_text.strip()) < 100:
                print(f"   WARNING: File seems empty or too short: {len(raw_text)} characters")
                print("   Skipping file (cannot read content)")
                failed_files += 1
                continue

            # Bước 2: Tạo law_id từ tên file
            law_id = generate_law_id(file_name)
            print(f"   Generated law_id: {law_id}")

            # Bước 3: Chunk document
            print("   Chunking document...")
            # Clean law_title (remove file extension)
            clean_law_title = file_name.replace('.docx', '').replace('.doc', '')

            chunks = chunk_law_document(
                text=raw_text,
                law_id=law_id,
                law_no=law_no,
                law_title=clean_law_title,
                issued_date=issued_date,
                effective_date=effective_date,
                expiry_date=None,
                signer=signer
            )

            if not chunks:
                print("   WARNING: No chunks created from file")
                failed_files += 1
                continue

            # Bước 4: Thêm metadata và chuẩn bị cho save
            print("   Preparing chunks...")
            for j, chunk in enumerate(chunks):
                # Sử dụng metadata đã có sẵn từ chunk_law_document (đã đúng format)
                all_chunks.append({
                    'id': chunk['id'],
                    'content': chunk['content'],
                    'metadata': chunk['metadata']
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

            print(f"   Successfully processed {len(chunks)} chunks")
            successful_chunks += len(chunks)

        except Exception as e:
            safe_error = str(e).encode('ascii', 'ignore').decode('ascii') or str(e)
            print(f"   ERROR: Error processing file: {safe_error}")
            failed_files += 1
            continue

    print(f"\nSummary:")
    print(f"   Successfully chunked: {len(law_file_paths) - failed_files} files")
    print(f"   Failed to process: {failed_files} files")
    print(f"   Total chunks created: {len(all_chunks)}")

    if all_chunks:
        print(f"   Average chunk length: {sum(len(c['content']) for c in all_chunks) // len(all_chunks):.0f} characters")

        # Thống kê theo category
        category_counts = {}
        for chunk in all_chunks:
            category = chunk.get('metadata', {}).get('source_category', 'Unknown')
            category_counts[category] = category_counts.get(category, 0) + 1

        print("\nDistribution by category:")
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
        "AI review sử dụng sampling thông minh để tránh vượt token limit.\n\n"
        "EXAMPLES:\n"
        "  python chunking.py --category BDS                    # Chunk BDS category\n"
        "  python chunking.py --file path/to/file.docx         # Chunk single file\n"
        "  python chunking.py --category BDS --AI --verbose    # Chunk + AI review với chi tiết\n"
        "  python chunking.py --file luat.docx --law-no '52/2014/QH13' --issued-date '2014-06-19' --signer 'Chủ tịch Quốc hội'  # Chunk với metadata đầy đủ\n"
        "  python chunking.py --validate --dry-run             # Test validation mà không ghi file\n\n"
        "WORKFLOW:\n"
        "  1. find_law_files.py  -> Tạo file paths theo category\n"
        "  2. chunking.py        -> Chunk documents thành chunks\n"
        "  3. main_refactored.py -> Evaluate và upload lên Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--AI", action="store_true", help="BẬT gọi Gemini để thẩm định chunks (mặc định KHÔNG gọi).")
    parser.add_argument("--sample-excerpts", type=int, default=2000, help="Tổng ký tự excerpts từ files gửi Gemini (mặc định 2000).")
    parser.add_argument("--max-chunks-sample", type=int, default=50, help="Số chunks tối đa sample để AI review (mặc định 50).")
    parser.add_argument("--max-files-sample", type=int, default=2, help="Số files tối đa lấy raw text để AI review (mặc định 2).")
    parser.add_argument("--strict-ok-only", action="store_true", help="Chỉ ghi chunks nếu Gemini trả 'ok' (chỉ khi --AI).")
    parser.add_argument("--api-key", help="Gemini API key (hoặc set GEMINI_API_KEY env var).")
    parser.add_argument("--category", help="Chunk theo category cụ thể (BDS, DN, TM, QDS). Nếu không chỉ định, chunk tất cả.")
    parser.add_argument("--file", help="Chunk một file cụ thể. Khi sử dụng --file, --category sẽ bị ignore.")
    parser.add_argument("--law-no", help="Số hiệu luật (mặc định rỗng)")
    parser.add_argument("--issued-date", help="Ngày ban hành luật (YYYY-MM-DD)")
    parser.add_argument("--effective-date", help="Ngày có hiệu lực (YYYY-MM-DD)")
    parser.add_argument("--signer", help="Người ký luật")
    parser.add_argument("--verbose", "-v", action="store_true", help="In chi tiết tiến trình chunking và AI review.")
    parser.add_argument("--validate", action="store_true", help="Validate chunks sau khi tạo (check format, metadata).")
    parser.add_argument("--dry-run", action="store_true", help="Test mode: chạy nhưng không ghi file output.")

    args = parser.parse_args()

    print("=" * 80)
    print("VIETNAMESE LAW DOCUMENT CHUNKING")
    if args.AI:
        print("WITH AI REVIEW ENABLED")
    if args.dry_run:
        print("DRY RUN MODE - No files will be written")
    if args.verbose:
        print("VERBOSE MODE - Detailed progress output")
    print("=" * 80)

    # Load .env if available
    if DOTENV_AVAILABLE:
        load_dotenv()

    # 1. Load danh sách file luật
    if args.file:
        safe_file = args.file.encode('ascii', 'ignore').decode('ascii') or args.file
        print(f"Chunking single file: {safe_file}")
        # Tạo file path info từ file cụ thể
        import os
        if os.path.exists(args.file):
            file_name = os.path.basename(args.file)
            # Tạo info tương tự như trong file paths
            law_file_paths = [{
                'path': args.file,
                'relative_path': args.file,  # Có thể cần điều chỉnh
                'category': 'Single File',
                'file_name': file_name,
                'extension': '.' + file_name.split('.')[-1] if '.' in file_name else ''
            }]
        else:
            print(f"ERROR: File not found: {args.file}")
            return
    elif args.category:
        print(f"Chunking category: {args.category}")
        law_file_paths = load_law_file_paths_by_category(args.category)
    else:
        print("Chunking all categories")
        law_file_paths = load_law_file_paths()

    if not law_file_paths:
        print("ERROR: Cannot proceed without law file paths!")
        if args.file:
            print(f"   File not found: {args.file}")
        elif args.category:
            print(f"   Make sure category '{args.category}' exists and find_law_files.py has been run")
        else:
            print("   Run find_law_files.py first")
        return

    # 2. Chunk tất cả documents
    if args.verbose:
        print(f"\nStarting to process {len(law_file_paths)} law documents...")

    all_chunks, summary = chunk_all_law_documents(
        law_file_paths, 
        law_no=args.law_no or "",
        issued_date=args.issued_date or "", 
        effective_date=args.effective_date or "",
        signer=args.signer or ""
    )

    if args.verbose:
        print(f"\nChunking completed: {len(all_chunks)} total chunks")
        print(f"   - Articles: {summary.get('articles', 0)}")
        print(f"   - Clauses: {summary.get('clauses', 0)}")
        print(f"   - Points: {summary.get('points', 0)}")

    # Validate chunks if requested
    if args.validate:
        validation = validate_chunks(all_chunks, args.verbose)
        if validation["invalid_chunks"] > 0:
            print(f"\nWARNING: Found {validation['invalid_chunks']} invalid chunks!")
            if not args.verbose:
                print(f"   Use --verbose to see details, or check validation summary")
        else:
            print(f"\nAll {validation['valid_chunks']} chunks are valid!")

    if args.dry_run:
        print(f"\nDRY RUN: Would save {len(all_chunks)} chunks to file")
        print("   Skipping file output (--dry-run enabled)")
        return

    # Thu thập raw text cho AI review (nếu bật AI)
    raw_texts = []
    if args.AI:
        print(f"\nCollecting raw texts for AI review (max {args.max_files_sample} files)...")
        files_sampled = 0
        import os  # Ensure os is available in this scope
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
                        safe_filename = file_info['file_name'].encode('ascii', 'ignore').decode('ascii') or file_info['file_name']
                        print(f"   Sampled file {files_sampled}: {safe_filename} ({len(raw_text)} chars)")
                except Exception as e:
                    safe_filename = file_info['file_name'].encode('ascii', 'ignore').decode('ascii') or file_info['file_name']
                    safe_error = str(e).encode('ascii', 'ignore').decode('ascii') or str(e)
                    print(f"   WARNING: Failed to read {safe_filename}: {safe_error}")
                    continue
        print(f"   Collected raw text excerpts from {len(raw_texts)} files (total ~{sum(len(t) for t in raw_texts)} chars)")

    if not all_chunks:
        print("Failed: No chunks were created!")
        return

    # 3. Save chunks ra JSON
    # Tạo tên file theo format: [PREFIX_]chunk_Giờphútgiây_ngàythángnăm(2 số cuối)
    now = datetime.now()
    timestamp = now.strftime("%H%M%S_%d%m%y")  # HHMMSS_DDMMYY

    if args.file:
        # Tạo prefix từ tên file (không có extension)
        file_name = os.path.basename(args.file)
        prefix = file_name.replace('.doc', '').replace('.docx', '').replace(' ', '_')[:30]  # Giới hạn độ dài
        chunks_json_path = f"data/{prefix}_chunk_{timestamp}.json"
    elif args.category:
        chunks_json_path = f"data/{args.category}_chunk_{timestamp}.json"
    else:
        chunks_json_path = f"data/chunk_{timestamp}.json"
    safe_path = chunks_json_path.encode('ascii', 'ignore').decode('ascii') or chunks_json_path
    print(f"\nSaving {len(all_chunks)} chunks to {safe_path}...")

    try:
        # Tạo thư mục data nếu chưa có
        Path("data").mkdir(exist_ok=True)

        save_chunks_to_json(all_chunks, chunks_json_path)
        safe_path = chunks_json_path.encode('ascii', 'ignore').decode('ascii') or chunks_json_path
        print(f"Successfully saved {len(all_chunks)} chunks to {safe_path}")
        print("   (Similar to hn2014_chunks.json format)")

    except Exception as e:
        print(f"ERROR: Error saving chunks: {e}")
        return

    # Nếu KHÔNG bật --AI: kết thúc tại đây
    if not args.AI:
        safe_path = chunks_json_path.encode('ascii', 'ignore').decode('ascii') or chunks_json_path
        print(f"\nCompleted successfully!")
        print(f"   Total chunks: {len(all_chunks)}")
        print(f"   Output file: {safe_path}")
        print(f"\nNext: Run main_refactored.py to evaluate with chunks from {safe_path}")
        return

    # ===== Khi --AI bật: gọi Gemini để thẩm định =====
    if args.verbose:
        print(f"\nPreparing AI review payload...")

    payload = build_review_payload(
        chunks=all_chunks,
        summary=summary,
        raw_texts=raw_texts,
        sample_excerpts_chars=args.sample_excerpts,
        max_chunks_sample=args.max_chunks_sample
    )

    print(f"\nAI: Calling GEMINI (gemini 2.0 Flash) for chunking review =====")
    print(f"   Sending: {len(payload['chunks_preview'])} sampled chunks + {len(raw_texts)} file excerpts")
    print(f"   Payload size estimate: ~{len(str(payload)) // 1000}KB")

    if args.verbose:
        print(f"   Summary stats: {payload['summary']}")
        print(f"   Excerpts length: {len(payload['excerpts'])} chars")
        print(f"   Note: {payload['note']}")
    try:
        review = call_gemini_review(payload, args.api_key)
    except Exception as e:
        print(f"ERROR: Failed to call Gemini: {e}", file=sys.stderr)
        if args.strict_ok_only:
            print("Failed: --strict-ok-only enabled, cannot proceed without AI review.")
            return
        else:
            safe_path = chunks_json_path.encode('ascii', 'ignore').decode('ascii') or chunks_json_path
            print(f"Warning: AI review failed. Chunks saved to: {safe_path}")
            print(f"\nCompleted (without AI review)!")
            print(f"   Total chunks: {len(all_chunks)}")
            print(f"   Output file: {safe_path}")
            return

    status = review.get("status", "issues_found")
    confidence = review.get("confidence", 0.0)
    issues = review.get("issues", []) or []
    notes = review.get("notes", "")

    print(f"- Trạng thái: {status} | Confidence: {confidence:.2f}")
    if notes:
        print(f"- Ghi chú: {notes[:4000]}")

    if issues:
        print("\nAI Issues Report:")
        for i, it in enumerate(issues, 1):
            print(f"{i:02d}. [{it.get('severity','?')}] ({it.get('category','other')}) "
                  f"{it.get('citation') or it.get('id') or ''}")
            print(f"    - {it.get('message','(no message)')}")
            if it.get('suggestion'):
                print(f"    -> Suggestion: {it['suggestion']}")
        issues_path = chunks_json_path.replace('.json', '.issues.json')
        with open(issues_path, 'w', encoding='utf-8') as f:
            json.dump(review, f, ensure_ascii=False, indent=2)
        print(f"\nSaved issues report: {issues_path}")

        if args.strict_ok_only:
            print("Failed: --strict-ok-only enabled, AI did not confirm OK.")
            return

    safe_path = chunks_json_path.encode('ascii', 'ignore').decode('ascii') or chunks_json_path
    print(f"\nCompleted successfully!")
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Output file: {safe_path}")
    if args.file:
        safe_basename = os.path.basename(args.file).encode('ascii', 'ignore').decode('ascii') or os.path.basename(args.file)
        print(f"   Source: Single file - {safe_basename}")
    elif args.category:
        print(f"   Source: Category - {args.category}")
    else:
        print(f"   Source: All categories")
    if args.AI and status == "ok":
        print("   AI confirmed: Chunking OK!")
    print(f"\nNext: Run main_refactored.py to evaluate with chunks from {safe_path}")

if __name__ == "__main__":
    main()
