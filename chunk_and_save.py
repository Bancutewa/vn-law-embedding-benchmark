#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chunk văn bản luật và lưu ra file JSON
Tái sử dụng được cho các file và category khác nhau với metadata đầy đủ
"""

import os
import json
import argparse
import pathlib
from pathlib import Path
from typing import List, Dict, Any

# Import từ modules
from modules.chunking import chunk_law_document
from modules.data_loader import save_chunks_to_json

# Import AI functions từ chunking.py gốc
import sys
import os
sys.path.append(os.path.dirname(__file__))
from chunking import build_review_payload, call_gemini_review

def read_docx_content(file_path: str) -> str:
    """Đọc nội dung file docx"""
    from modules.chunking import read_docx
    return read_docx(file_path)

def chunk_single_file(file_path: str, metadata: Dict[str, str], verbose: bool = False) -> List[Dict[str, Any]]:
    """Chunk một file đơn lẻ với metadata đầy đủ"""
    if verbose:
        safe_file_path = file_path.encode('ascii', 'ignore').decode('ascii') or file_path
        print(f"Reading file: {safe_file_path}")

    # Đọc nội dung file
    raw_text = read_docx_content(file_path)
    if not raw_text:
        raise ValueError(f"Could not read content from {file_path}")

    if verbose:
        print(f"   Read {len(raw_text):,} characters")

    # Chunk document
    if verbose:
        print("Chunking document...")

    chunks = chunk_law_document(
        text=raw_text,
        law_id=metadata.get('law_id', 'LAW'),
        law_no=metadata.get('law_no', ''),
        law_title=metadata.get('law_title', ''),
        issued_date=metadata.get('issued_date', ''),
        effective_date=metadata.get('effective_date', ''),
        expiry_date=None,
        signer=metadata.get('signer', '')
    )

    if verbose:
        print(f"   Created {len(chunks)} chunks")

    return chunks

def generate_output_filename(file_path: str, category: str = "", custom_name: str = "") -> str:
    """Tạo tên file output tự động"""
    from datetime import datetime

    if custom_name:
        base_name = custom_name
    elif file_path:
        # Lấy tên file không có extension
        file_name = Path(file_path).stem
        # Clean tên file (loại bỏ khoảng trắng và ký tự đặc biệt)
        base_name = "".join(c for c in file_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        base_name = base_name.replace(' ', '_')[:30]  # Giới hạn độ dài
    else:
        base_name = category or "chunk"

    timestamp = datetime.now().strftime("%H%M%S_%d%m%y")
    return f"{base_name}_chunk_{timestamp}.json"

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Chunk văn bản luật và lưu ra JSON với metadata đầy đủ",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ========== THAM SỐ CẦN SỬA ==========
    # File input - sửa path này khi cần
    parser.add_argument(
        "--file",
        default="law_content/Bất động sản/Luật xây dựng/Luật_/43_VBHN-VPQH_no_refs.docx",  # ←←← SỬA PATH FILE Ở ĐÂY
        help="Path to input docx file"
    )
    # Category (cho naming)
    parser.add_argument(
        "--category",
        default="BDS",  # ←←← SỬA CATEGORY Ở ĐÂY
        help="Category name for naming"
    )

    # ========== METADATA CẦN SỬA ==========
    # Thông tin luật pháp - sửa các thông tin này khi cần
    parser.add_argument(
        "--law-no",
        default="50/2014/QH13 ",  # ←←← SỬA SỐ HIỆU LUẬT Ở ĐÂY
        help="Law number"
    )

    parser.add_argument(
        "--law-title",
        default="Luật Xây dựng",  # ←←← SỬA TÊN LUẬT Ở ĐÂY
        help="Law title"
    )

    parser.add_argument(
        "--law-id",
        default="LXAYDUNG",  # ←←← SỬA LAW ID Ở ĐÂY
        help="Law ID"
    )

    parser.add_argument(
        "--issued-date",
        default="2014-6-18",  # ←←← SỬA NGÀY BAN HÀNH Ở ĐÂY
        help="Issued date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--effective-date",
        default="2015-1-1",  # ←←← SỬA NGÀY CÓ HIỆU LỰC Ở ĐÂY
        help="Effective date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--signer",
        default="",  # ←←← SỬA NGƯỜI KÝ Ở ĐÂY
        help="Signer name"
    )

    # ========== TÙY CHỌN OUTPUT ==========
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory (default: data)"
    )

    parser.add_argument(
        "--output-name",
        help="Custom output filename (without extension). If not specified, auto-generated"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if exists"
    )

    # ========== TÙY CHỌN CHẠY ==========
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test mode: chỉ chunk và validate, không lưu file"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate chunks sau khi tạo"
    )

    # ========== AI REVIEW OPTIONS ==========
    parser.add_argument(
        "--AI",
        action="store_true",
        help="Bật Gemini AI thẩm định chunks (default: OFF)"
    )

    parser.add_argument(
        "--max-files-sample",
        type=int,
        default=2,
        help="Số files tối đa lấy raw text cho AI review (default: 2)"
    )

    parser.add_argument(
        "--max-chunks-sample",
        type=int,
        default=50,
        help="Số chunks tối đa gửi AI review (default: 50)"
    )

    parser.add_argument(
        "--sample-excerpts",
        type=int,
        default=2000,
        help="Tổng ký tự excerpts từ files gửi AI (default: 2000)"
    )

    parser.add_argument(
        "--api-key",
        help="Gemini API key (hoặc set GEMINI_API_KEY env var)"
    )

    parser.add_argument(
        "--strict-ok-only",
        action="store_true",
        help="Chỉ ghi chunks nếu AI trả 'ok' (chỉ khi --AI)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("VIETNAMESE LAW DOCUMENT CHUNKING & SAVE")
    print("=" * 80)
    safe_file = args.file.encode('ascii', 'ignore').decode('ascii') or args.file
    safe_law_title = args.law_title.encode('ascii', 'ignore').decode('ascii') or args.law_title
    safe_signer = args.signer.encode('ascii', 'ignore').decode('ascii') or args.signer

    print(f"Input file: {safe_file}")
    print(f"Category: {args.category}")
    print(f"Law: {safe_law_title} ({args.law_no})")
    print(f"Law ID: {args.law_id}")
    print(f"Issued: {args.issued_date}")
    print(f"Effective: {args.effective_date}")
    print(f"Signer: {safe_signer}")
    print(f"Output dir: {args.output_dir}")

    if args.AI:
        print(f"AI Review: ENABLED")
        print(f"  Max files sample: {args.max_files_sample}")
        print(f"  Max chunks sample: {args.max_chunks_sample}")
        print(f"  Sample excerpts: {args.sample_excerpts} chars")
        if args.strict_ok_only:
            print("  Strict OK only: ENABLED")
    else:
        print("AI Review: DISABLED")

    if args.output_name:
        output_filename = f"{args.output_name}.json"
    else:
        output_filename = generate_output_filename(args.file, args.category)

    output_path = os.path.join(args.output_dir, output_filename)
    safe_output_path = output_path.encode('ascii', 'ignore').decode('ascii') or output_path
    print(f"Output file: {safe_output_path}")

    if args.dry_run:
        print("DRY RUN MODE - No file will be saved")
    print("=" * 80)

    try:
        # 1. Kiểm tra file input
        if not os.path.exists(args.file):
            safe_file = args.file.encode('ascii', 'ignore').decode('ascii') or args.file
            print(f"ERROR: Input file not found: {safe_file}")
            return

        # 2. Chuẩn bị metadata
        metadata = {
            'law_id': args.law_id,
            'law_no': args.law_no,
            'law_title': args.law_title,
            'issued_date': args.issued_date,
            'effective_date': args.effective_date,
            'signer': args.signer
        }

        # 3. Chunk document
        chunks = chunk_single_file(args.file, metadata, args.verbose)

        if not chunks:
            print("ERROR: No chunks were created!")
            return

        # 4. Validate chunks nếu được yêu cầu
        if args.validate:
            from chunking import validate_chunks
            validation = validate_chunks(chunks, args.verbose)
            if validation["invalid_chunks"] > 0:
                print(f"WARNING: Found {validation['invalid_chunks']} invalid chunks!")
                if not args.verbose:
                    print("   Use --verbose to see details")
            else:
                print("All chunks are valid!")

        # 5. AI Review nếu được bật
        if args.AI:
            print("AI: Preparing AI review payload...")
            # Thu thập raw text cho AI review (chỉ lấy file hiện tại)
            raw_texts = []
            if args.max_files_sample > 0:
                try:
                    raw_text = read_docx_content(args.file)
                    if raw_text and len(raw_text.strip()) > 100:
                        # Giới hạn ký tự mỗi file
                        limited_text = raw_text[:5000]  # Giới hạn 5k chars mỗi file
                        raw_texts.append(limited_text)
                        safe_file = args.file.encode('ascii', 'ignore').decode('ascii') or args.file
                        print(f"   Sampled file: {safe_file} ({len(raw_text)} chars)")
                except Exception as e:
                    safe_file = args.file.encode('ascii', 'ignore').decode('ascii') or args.file
                    safe_error = str(e).encode('ascii', 'ignore').decode('ascii') or str(e)
                    print(f"   WARNING: Failed to read {safe_file}: {safe_error}")

            # Chuẩn bị summary cho AI review
            summary = {
                "chapters_seen": [],  # Không có thông tin chapters trong single file processing
                "articles": sum(1 for c in chunks if c.get('metadata', {}).get('article_no') and not c.get('metadata', {}).get('clause_no')),
                "article_intro": sum(1 for c in chunks if 'intro' in c.get('id', '').lower()),
                "clauses": sum(1 for c in chunks if c.get('metadata', {}).get('clause_no') and not c.get('metadata', {}).get('point_letter')),
                "points": sum(1 for c in chunks if c.get('metadata', {}).get('point_letter')),
                "citations": [c.get('metadata', {}).get('exact_citation', '') for c in chunks if c.get('metadata', {}).get('exact_citation')],
                "warnings": [],
                "halted_reason": None,
                "total_chunks": len(chunks)
            }

            print("AI: Calling Gemini for chunking review...")
            print(f"   Sending: {min(args.max_chunks_sample, len(chunks))} sampled chunks + {len(raw_texts)} file excerpts")
            print(f"   Payload size estimate: ~{len(str({'summary': summary, 'excerpts': raw_texts[0][:args.sample_excerpts] if raw_texts else '', 'chunks_preview': chunks[:args.max_chunks_sample]})) // 1000}KB")

            payload = build_review_payload(
                chunks=chunks,
                summary=summary,
                raw_texts=raw_texts,
                sample_excerpts_chars=args.sample_excerpts,
                max_chunks_sample=args.max_chunks_sample
            )

            try:
                review = call_gemini_review(payload, args.api_key)
                status = review.get("status", "issues_found")
                confidence = review.get("confidence", 0.0)
                issues = review.get("issues", []) or []
                notes = review.get("notes", "")

                print(f"- Status: {status} | Confidence: {confidence:.2f}")
                if notes:
                    print(f"- Notes: {notes[:1000]}...")

                if issues:
                    print("AI Issues Report:")
                    for i, it in enumerate(issues, 1):
                        citation = it.get('citation') or it.get('id') or ''
                        print(f"{i:02d}. [{it.get('severity','?')}] ({it.get('category','other')}) {citation}")
                        print(f"    - {it.get('message','(no message)')}")
                        if it.get('suggestion'):
                            print(f"    -> Suggestion: {it['suggestion']}")

                    # Lưu issues report
                    issues_path = output_path.replace('.json', '.issues.json')
                    safe_issues_path = issues_path.encode('ascii', 'ignore').decode('ascii') or issues_path
                    try:
                        import json
                        with open(issues_path, 'w', encoding='utf-8') as f:
                            json.dump(review, f, ensure_ascii=False, indent=2)
                        print(f"Saved issues report: {safe_issues_path}")
                    except Exception as e:
                        print(f"Warning: Could not save issues report: {e}")

                    if args.strict_ok_only:
                        print("Failed: --strict-ok-only enabled, AI did not confirm OK.")
                        return
                else:
                    print("AI confirmed: Chunking OK!")

            except Exception as e:
                safe_error = str(e).encode('ascii', 'ignore').decode('ascii') or str(e)
                print(f"ERROR: Failed to call Gemini: {safe_error}")
                if args.strict_ok_only:
                    print("Failed: --strict-ok-only enabled, cannot proceed without AI review.")
                    return
                else:
                    print("Warning: AI review failed. Proceeding without AI confirmation.")

        # 7. Kiểm tra file output
        if not args.dry_run:
            if os.path.exists(output_path) and not args.overwrite:
                safe_output_path = output_path.encode('ascii', 'ignore').decode('ascii') or output_path
                print(f"WARNING: Output file already exists: {safe_output_path}")
                choice = input("   Overwrite? (y/N): ").strip().lower()
                if choice != 'y':
                    print("Aborted")
                    return

        # 8. Tạo thư mục output nếu cần
        if not args.dry_run:
            os.makedirs(args.output_dir, exist_ok=True)

        # 9. Lưu chunks ra file
        if args.dry_run:
            print("DRY RUN: Skipping file save")
        else:
            safe_output_path = output_path.encode('ascii', 'ignore').decode('ascii') or output_path
            print(f"Saving {len(chunks)} chunks to {safe_output_path}...")
            save_chunks_to_json(chunks, output_path)
            print(f"Successfully saved {len(chunks)} chunks")

        # 10. Thông tin tổng kết
        print("\nCHUNKING SUMMARY:")
        safe_input_file = os.path.basename(args.file).encode('ascii', 'ignore').decode('ascii') or os.path.basename(args.file)
        safe_law_title = args.law_title.encode('ascii', 'ignore').decode('ascii') or args.law_title
        print(f"   Input file: {safe_input_file}")
        print(f"   Law: {safe_law_title}")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Avg length: {sum(len(c['content']) for c in chunks) // len(chunks):.1f} chars")
        # Thống kê theo loại chunk
        article_count = sum(1 for c in chunks if c.get('metadata', {}).get('article_no') and not c.get('metadata', {}).get('clause_no'))
        clause_count = sum(1 for c in chunks if c.get('metadata', {}).get('clause_no') and not c.get('metadata', {}).get('point_letter'))
        point_count = sum(1 for c in chunks if c.get('metadata', {}).get('point_letter'))

        print("   Breakdown:")
        print(f"      - Articles: {article_count}")
        print(f"      - Clauses: {clause_count}")
        print(f"      - Points: {point_count}")

        if not args.dry_run:
            safe_output_path = output_path.encode('ascii', 'ignore').decode('ascii') or output_path
            print(f"\nSUCCESS! File saved to: {safe_output_path}")
            print("\nNEXT STEPS:")
            safe_output_path_cmd = output_path.replace('\\', '\\\\')  # Escape backslashes for Windows
            print(f"   1. Review chunks: python -m json.tool \"{safe_output_path}\" | head -50")
            print(f"   2. Embed & upload: python embed_and_upload.py --chunk-file \"{safe_output_path}\" --category {args.category}")
            print(f"   3. Run evaluation: python main_refactored.py")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
