#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chunk văn bản luật và lưu ra file JSON
Tái sử dụng được cho các file và category khác nhau với metadata đầy đủ

HƯỚNG DẪN SỬ DỤNG NHANH:
========================

Cách 1: Config files ngay trong script
    - Sửa DEFAULT_FILES ở dòng 38-40
    - Chạy: python chunk_and_save.py
    - Output: data/{CATEGORY}/...

Cách 2: Command line với 1 file
    python chunk_and_save.py --file "path/to/law.docx" --category BDS

Cách 3: Command line với nhiều files
    python chunk_and_save.py --files "law1.docx" "law2.docx" --category BDS

Cách 4: AI Review (khuyến nghị)
    python chunk_and_save.py --AI --verbose

OUTPUT:
    Files được lưu vào: data/{CATEGORY}/
    Ví dụ: data/BDS/Luat_1_chunk_143052_141025.json
"""

import os
import json
import argparse
import pathlib
from pathlib import Path
from typing import List, Dict, Any


DEFAULT_FILES = [
    # "law_content/Bất động sản/Luật Kinh Doanh Bất Động Sản/Văn bản pháp luật_/Luật Kinh doanh bất động sản 2023 .docx",
    # "law_content/Bất động sản/Luật Nhà ở/Văn bản pháp luật_/Luật Nhà ở 2023.docx",
    # "law_content/Bất động sản/Luật Đất đai/Văn bản pháp luật_/Luật Đất đai 2024.docx",
    # "law_content/Bất động sản/Luật thuế sử dụng đất phi nông nghiệp_/Văn bản pháp luật/LUẬT THUẾ SỬ DỤNG ĐẤT PHI NÔNG NGHIỆP.docx"
    "law_content/Quyền dân sự_/Luật Hôn Nhân Gia Đình/Văn bản pháp luật_/luat_hon_nhan_gia_dinh_2014.docx",  

]

# ==================================================================

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

def chunk_single_file(file_path: str, metadata: Dict[str, str], verbose: bool = False):
    """Chunk một file đơn lẻ với metadata đầy đủ
    
    Returns:
        tuple: (chunks, chunk_stats) - List of chunks và dict stats
    """
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

    chunks, chunk_stats = chunk_law_document(
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
        print(f"   Stats - Articles: {chunk_stats.get('articles', 0)}, Article Intro: {chunk_stats.get('article_intro', 0)}, Clauses: {chunk_stats.get('clauses', 0)}, Points: {chunk_stats.get('points', 0)}")

    return chunks, chunk_stats

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
    # File input - CÓ 3 CÁCH:
    # 1. Sửa DEFAULT_FILES ở đầu file  
    # 2. Dùng --file cho 1 file
    # 3. Dùng --files cho nhiều files
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--file",
        # default="law_content/Bất động sản/Luật Kinh Doanh Bất Động Sản/Văn bản pháp luật_/Luật Kinh doanh bất động sản 2023 .docx",  # ←←← SỬA PATH FILE Ở ĐÂY
        help="Path to single input docx file (override DEFAULT_FILES)"
    )
    input_group.add_argument(
        "--files",
        nargs="+",
        # default="law_content/Bất động sản/Luật Kinh Doanh Bất Động Sản/Văn bản pháp luật_/Luật Kinh doanh bất động sản 2023 .docx",  # ←←← SỬA PATH FILE Ở ĐÂY
        help="Multiple docx files to chunk (override DEFAULT_FILES)"
    )
    # Category (cho naming và subfolder)
    parser.add_argument(
        "--category",
        default="LHN2014",  # ←←← SỬA CATEGORY Ở ĐÂY (hoặc dùng --category khi chạy)
        help="Category name for naming and output subfolder (e.g. BDS, QDS, TM)"
    )

    # ========== METADATA CẦN SỬA ==========
    # Thông tin luật pháp - sửa các thông tin này khi cần
    parser.add_argument(
        "--law-no",
        default=" ",  # ←←← SỬA SỐ HIỆU LUẬT Ở ĐÂY
        help="Law number"
    )

    parser.add_argument(
        "--law-title",
        default="LUẬT HÔN NHÂN GIA ĐÌNH",  # ←←← SỬA TÊN LUẬT Ở ĐÂY
        help="Law title"
    )

    parser.add_argument(
        "--law-id",
        default="LHN2014",  # ←←← SỬA LAW ID Ở ĐÂY
        help="Law ID"
    )

    parser.add_argument(
        "--issued-date",
        default="",  # ←←← SỬA NGÀY BAN HÀNH Ở ĐÂY
        help="Issued date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--effective-date",
        default="",  # ←←← SỬA NGÀY CÓ HIỆU LỰC Ở ĐÂY
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
        help="Output directory. Sẽ tự động tạo subfolder theo category (default: data)"
    )
    
    parser.add_argument(
        "--no-category-folder",
        action="store_true",
        help="Không tạo subfolder theo category (save trực tiếp vào output-dir)"
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

    # Xác định list files để chunk
    if args.file:
        input_files = [args.file]
    elif args.files:
        input_files = args.files
    elif DEFAULT_FILES:
        # Sử dụng DEFAULT_FILES từ config nếu không có --file hoặc --files
        input_files = DEFAULT_FILES
        print("📌 Using DEFAULT_FILES from config")
    else:
        print("ERROR: Please specify --file, --files, or configure DEFAULT_FILES in the script")
        print("Example:")
        print('  python chunk_and_save.py --file "path/to/law.docx"')
        print('  python chunk_and_save.py --files "path/to/law1.docx" "path/to/law2.docx"')
        print("Or edit DEFAULT_FILES in the script header")
        return

    # Xác định output directory (có hoặc không có category subfolder)
    if args.no_category_folder:
        final_output_dir = args.output_dir
    else:
        final_output_dir = os.path.join(args.output_dir, args.category)

    print("=" * 80)
    print("VIETNAMESE LAW DOCUMENT CHUNKING & SAVE")
    print("=" * 80)
    print(f"Input files: {len(input_files)} file(s)")
    for i, f in enumerate(input_files, 1):
        safe_f = os.path.basename(f).encode('ascii', 'ignore').decode('ascii') or os.path.basename(f)
        print(f"  {i}. {safe_f}")
    
    safe_law_title = args.law_title.encode('ascii', 'ignore').decode('ascii') or args.law_title
    safe_signer = args.signer.encode('ascii', 'ignore').decode('ascii') or args.signer

    print(f"Category: {args.category}")
    print(f"Law: {safe_law_title} ({args.law_no})")
    print(f"Law ID: {args.law_id}")
    print(f"Issued: {args.issued_date}")
    print(f"Effective: {args.effective_date}")
    print(f"Signer: {safe_signer}")
    print(f"Output dir: {final_output_dir}")

    if args.AI:
        print(f"AI Review: ENABLED")
        print(f"  Max files sample: {args.max_files_sample}")
        print(f"  Max chunks sample: {args.max_chunks_sample}")
        print(f"  Sample excerpts: {args.sample_excerpts} chars")
        if args.strict_ok_only:
            print("  Strict OK only: ENABLED")
    else:
        print("AI Review: DISABLED")

    if args.dry_run:
        print("DRY RUN MODE - No file will be saved")
    print("=" * 80)

    # Tạo output directory nếu chưa tồn tại
    os.makedirs(final_output_dir, exist_ok=True)

    # Stats tổng
    total_chunks = 0
    total_successful = 0
    total_failed = 0
    all_output_files = []

    # Loop qua tất cả input files
    for file_idx, input_file in enumerate(input_files, 1):
        print(f"\n{'=' * 80}")
        print(f"Processing file {file_idx}/{len(input_files)}")
        print(f"{'=' * 80}")
        
        safe_file = os.path.basename(input_file).encode('ascii', 'ignore').decode('ascii') or os.path.basename(input_file)
        print(f"File: {safe_file}")

        try:
            # 1. Kiểm tra file input
            if not os.path.exists(input_file):
                safe_file = input_file.encode('ascii', 'ignore').decode('ascii') or input_file
                print(f"ERROR: Input file not found: {safe_file}")
                total_failed += 1
                continue

            # 2. Generate output filename
            if args.output_name and len(input_files) == 1:
                output_filename = f"{args.output_name}.json"
            else:
                output_filename = generate_output_filename(input_file, args.category)
            
            output_path = os.path.join(final_output_dir, output_filename)
            safe_output_path = output_path.encode('ascii', 'ignore').decode('ascii') or output_path
            print(f"Output: {safe_output_path}")

            # 3. Chuẩn bị metadata
            metadata = {
                'law_id': args.law_id,
                'law_no': args.law_no,
                'law_title': args.law_title,
                'issued_date': args.issued_date,
                'effective_date': args.effective_date,
                'signer': args.signer
            }

            # 4. Chunk document
            chunks, chunk_stats = chunk_single_file(input_file, metadata, args.verbose)

            if not chunks:
                print("ERROR: No chunks were created!")
                total_failed += 1
                continue

            # 5. Validate chunks nếu được yêu cầu
            if args.validate:
                from chunking import validate_chunks
                validation = validate_chunks(chunks, args.verbose)
                if validation["invalid_chunks"] > 0:
                    print(f"WARNING: Found {validation['invalid_chunks']} invalid chunks!")
                    if not args.verbose:
                        print("   Use --verbose to see details")
                else:
                    print("All chunks are valid!")

            # 6. AI Review nếu được bật
            if args.AI:
                print("AI: Preparing AI review payload...")
                raw_texts = []
                if args.max_files_sample > 0:
                    try:
                        raw_text = read_docx_content(input_file)
                        if raw_text and len(raw_text.strip()) > 100:
                            limited_text = raw_text[:5000]
                            raw_texts.append(limited_text)
                            safe_file_name = os.path.basename(input_file).encode('ascii', 'ignore').decode('ascii') or os.path.basename(input_file)
                            print(f"   Sampled file: {safe_file_name} ({len(raw_text)} chars)")
                    except Exception as e:
                        safe_file_name = os.path.basename(input_file).encode('ascii', 'ignore').decode('ascii') or os.path.basename(input_file)
                        safe_error = str(e).encode('ascii', 'ignore').decode('ascii') or str(e)
                        print(f"   WARNING: Failed to read {safe_file_name}: {safe_error}")

                summary = {
                    "chapters_seen": chunk_stats.get('chapters_seen', []),
                    "articles": chunk_stats.get('articles', 0),
                    "article_intro": chunk_stats.get('article_intro', 0),
                    "clauses": chunk_stats.get('clauses', 0),
                    "points": chunk_stats.get('points', 0),
                    "citations": chunk_stats.get('citations', []),
                    "warnings": [],
                    "halted_reason": None,
                    "total_chunks": chunk_stats.get('total_chunks', len(chunks))
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
                            total_failed += 1
                            continue
                    else:
                        print("AI confirmed: Chunking OK!")

                except Exception as e:
                    safe_error = str(e).encode('ascii', 'ignore').decode('ascii') or str(e)
                    print(f"ERROR: Failed to call Gemini: {safe_error}")
                    if args.strict_ok_only:
                        print("Failed: --strict-ok-only enabled, cannot proceed without AI review.")
                        total_failed += 1
                        continue
                    else:
                        print("Warning: AI review failed. Proceeding without AI confirmation.")

            # 7. Kiểm tra file output
            if not args.dry_run:
                if os.path.exists(output_path) and not args.overwrite:
                    safe_output_path = output_path.encode('ascii', 'ignore').decode('ascii') or output_path
                    print(f"WARNING: Output file already exists: {safe_output_path}")
                    choice = input("   Overwrite? (y/N): ").strip().lower()
                    if choice != 'y':
                        print("Skipped")
                        total_failed += 1
                        continue

            # 8. Lưu chunks ra file
            if args.dry_run:
                print("DRY RUN: Skipping file save")
            else:
                safe_output_path = output_path.encode('ascii', 'ignore').decode('ascii') or output_path
                print(f"Saving {len(chunks)} chunks to {safe_output_path}...")
                save_chunks_to_json(chunks, output_path)
                print(f"✅ Successfully saved {len(chunks)} chunks")
                all_output_files.append(output_path)

            # 9. File summary
            total_chunks += len(chunks)
            total_successful += 1
            
            print(f"\n📊 File Summary:")
            safe_input_file = os.path.basename(input_file).encode('ascii', 'ignore').decode('ascii') or os.path.basename(input_file)
            print(f"   File: {safe_input_file}")
            print(f"   Chunks: {len(chunks)}")
            if chunks:
                print(f"   Avg length: {sum(len(c['content']) for c in chunks) // len(chunks):.1f} chars")
            print(f"   Articles: {chunk_stats.get('articles', 0)}")
            print(f"   Clauses: {chunk_stats.get('clauses', 0)}")
            print(f"   Points: {chunk_stats.get('points', 0)}")

        except Exception as e:
            print(f"❌ ERROR processing file: {e}")
            import traceback
            traceback.print_exc()
            total_failed += 1

    # TỔNG KẾT CUỐI CÙNG
    print(f"\n{'=' * 80}")
    print("🎉 BATCH CHUNKING COMPLETED!")
    print(f"{'=' * 80}")
    print(f"Total files processed: {len(input_files)}")
    print(f"  ✅ Successful: {total_successful}")
    print(f"  ❌ Failed: {total_failed}")
    print(f"Total chunks created: {total_chunks}")
    
    if all_output_files:
        print(f"\n📁 Output files ({len(all_output_files)}):")
        for f in all_output_files:
            safe_f = f.encode('ascii', 'ignore').decode('ascii') or f
            print(f"  - {safe_f}")
        
        print(f"\n📂 Output directory: {final_output_dir}")
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Review chunks trong folder: {final_output_dir}")
        print(f"   2. Batch upload: python batch_embed_upload.py --directory \"{final_output_dir}\" --category {args.category}")
        print(f"   3. Hoặc merge: python merge_chunks.py --directory \"{final_output_dir}\" --category {args.category}")

if __name__ == "__main__":
    main()
