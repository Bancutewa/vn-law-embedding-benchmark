#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge nhiều chunk JSON files thành 1 file lớn
Useful khi muốn upload 1 lần thay vì nhiều lần

HƯỚNG DẪN SỬ DỤNG NHANH:
========================

Cách 1: Config ngay trong script 
    - Sửa DEFAULT_DIRECTORY và DEFAULT_CATEGORY ở dòng 36-37
    - Chạy: python merge_chunks.py
    - Output: data/{CATEGORY}_merged_chunk_*.json

Cách 2: Command line với directory
    python merge_chunks.py --directory data/BDS --category BDS

Cách 3: Command line với files cụ thể
    python merge_chunks.py --files "file1.json" "file2.json" --category BDS

OUTPUT:
    Merged file: data/{CATEGORY}_merged_chunk_HHMMSS_DDMMYY.json
"""

import os
import json
import argparse
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# ==================== CẤU HÌNH MERGE ====================
# 🔧 HƯỚNG DẪN SỬ DỤNG:
# 1. Để merge files trong 1 folder: Uncomment và sửa DEFAULT_DIRECTORY
# 2. Để merge files cụ thể: Uncomment và sửa DEFAULT_FILES
# 3. Để dùng command line: Để cả 2 = None

# VÍ DỤ 1: Merge tất cả files trong folder (RECOMMENDED)
DEFAULT_DIRECTORY = "data/BDS"  # "data/BDS"  # ← Uncomment và sửa path
DEFAULT_CATEGORY = "BDS"  # ← Category cho naming output file
DEFAULT_PATTERN = "*_chunk_*.json"  # Pattern để tìm files

# VÍ DỤ 2: Merge các files cụ thể
DEFAULT_FILES = None  # ["data/BDS/file1.json", "data/BDS/file2.json"]

# 💡 TIP: Sau khi config xong, chỉ cần chạy: python merge_chunks.py
# ==================================================================


def load_chunks_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load chunks từ 1 file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    return chunks


def merge_chunk_files(file_paths: List[str], remove_duplicates: bool = True) -> List[Dict[str, Any]]:
    """Merge nhiều chunk files thành 1 list"""
    all_chunks = []
    seen_ids = set()
    
    print(f"📦 Merging {len(file_paths)} files...")
    
    for i, file_path in enumerate(file_paths, 1):
        basename = os.path.basename(file_path)
        print(f"\n[{i}/{len(file_paths)}] Loading: {basename}")
        
        try:
            chunks = load_chunks_from_file(file_path)
            
            if remove_duplicates:
                # Remove duplicates by chunk ID
                new_chunks = []
                duplicates = 0
                for chunk in chunks:
                    chunk_id = chunk.get('id')
                    if chunk_id not in seen_ids:
                        new_chunks.append(chunk)
                        seen_ids.add(chunk_id)
                    else:
                        duplicates += 1
                
                all_chunks.extend(new_chunks)
                print(f"   ✅ Added {len(new_chunks)} chunks ({duplicates} duplicates skipped)")
            else:
                all_chunks.extend(chunks)
                print(f"   ✅ Added {len(chunks)} chunks")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    print(f"\n✅ Total chunks after merge: {len(all_chunks)}")
    return all_chunks


def save_merged_chunks(chunks: List[Dict[str, Any]], output_path: str):
    """Save merged chunks ra file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved to: {output_path}")


def generate_output_filename(category: str = "", custom_name: str = "") -> str:
    """Tạo tên file output"""
    timestamp = datetime.now().strftime("%H%M%S_%d%m%y")
    
    if custom_name:
        base_name = custom_name
    elif category:
        base_name = f"{category}_merged"
    else:
        base_name = "merged"
    
    return f"{base_name}_chunk_{timestamp}.json"


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Merge nhiều chunk JSON files thành 1 file lớn",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Merge tất cả files trong thư mục data/
  python merge_chunks.py --directory data/ --output data/merged_all.json
  
  # Merge các files cụ thể
  python merge_chunks.py --files data/luat1.json data/luat2.json --output data/merged.json
  
  # Merge với pattern
  python merge_chunks.py --directory data/ --pattern "BDS_*_chunk_*.json" --output data/BDS_merged.json
  
  # Auto-generate output filename
  python merge_chunks.py --directory data/ --category BDS
        """
    )

    # Input options - CÓ 3 CÁCH:
    # 1. Sửa DEFAULT_DIRECTORY/DEFAULT_FILES ở đầu file (recommended) ⭐
    # 2. Dùng --directory cho folder
    # 3. Dùng --files cho files cụ thể
    input_group = parser.add_mutually_exclusive_group()  # Bỏ required=True để có thể dùng DEFAULT
    input_group.add_argument(
        "--directory",
        help="Thư mục chứa chunk files (override DEFAULT_DIRECTORY)"
    )
    input_group.add_argument(
        "--files",
        nargs="+",
        help="Danh sách files cụ thể (override DEFAULT_FILES)"
    )

    parser.add_argument(
        "--pattern",
        default="*_chunk_*.json",
        help="Pattern để tìm files (với --directory). Default: *_chunk_*.json"
    )

    parser.add_argument(
        "--output",
        help="Output file path. Nếu không chỉ định, sẽ auto-generate"
    )

    parser.add_argument(
        "--category",
        help="Category name (dùng cho auto-generate output filename)"
    )

    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory (default: data)"
    )

    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Giữ lại chunks duplicate (mặc định: remove duplicates by ID)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test mode: không ghi file output"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("📦 MERGE CHUNK FILES")
    print("=" * 80)

    # 1. Xác định input source và category
    directory = None
    chunk_files = None
    pattern = args.pattern
    category = args.category
    
    if args.directory:
        # Command line --directory
        directory = args.directory
    elif args.files:
        # Command line --files
        chunk_files = args.files
    elif DEFAULT_DIRECTORY:
        # Config DEFAULT_DIRECTORY
        directory = DEFAULT_DIRECTORY
        category = category or DEFAULT_CATEGORY
        print("📌 Using DEFAULT_DIRECTORY from config")
    elif DEFAULT_FILES:
        # Config DEFAULT_FILES
        chunk_files = DEFAULT_FILES
        category = category or DEFAULT_CATEGORY
        print("📌 Using DEFAULT_FILES from config")
    else:
        print("❌ ERROR: Please specify input source!")
        print("Options:")
        print("  1. Config DEFAULT_DIRECTORY or DEFAULT_FILES in script header")
        print("  2. Use --directory: python merge_chunks.py --directory data/BDS")
        print("  3. Use --files: python merge_chunks.py --files file1.json file2.json")
        return

    # 2. Tìm files từ directory (nếu có)
    if directory:
        print(f"📁 Searching in: {directory}")
        print(f"   Pattern: {pattern}")
        chunk_files = glob.glob(os.path.join(directory, pattern))
        chunk_files.sort()

    if not chunk_files:
        print("❌ No chunk files found!")
        return

    print(f"✅ Found {len(chunk_files)} files:")
    for i, f in enumerate(chunk_files, 1):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"   {i}. {os.path.basename(f)} ({size_mb:.2f} MB)")

    # 3. Merge
    remove_dups = not args.keep_duplicates
    all_chunks = merge_chunk_files(chunk_files, remove_duplicates=remove_dups)

    if not all_chunks:
        print("❌ No chunks to save!")
        return

    # 4. Generate output path
    if args.output:
        output_path = args.output
    else:
        # Sử dụng category từ args hoặc DEFAULT_CATEGORY
        final_category = category or ""
        output_filename = generate_output_filename(
            category=final_category,
            custom_name=""
        )
        output_path = os.path.join(args.output_dir, output_filename)

    # 5. Save
    if args.dry_run:
        print(f"\n🔍 DRY RUN: Would save {len(all_chunks)} chunks to: {output_path}")
    else:
        # Tạo thư mục nếu cần
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        save_merged_chunks(all_chunks, output_path)
        
        # Stats
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n✅ SUCCESS!")
        print(f"   Total chunks: {len(all_chunks)}")
        print(f"   File size: {file_size_mb:.2f} MB")
        print(f"   Output: {output_path}")
        
        # Breakdown by law_id
        law_ids = {}
        for chunk in all_chunks:
            law_id = chunk.get('metadata', {}).get('law_id', 'Unknown')
            law_ids[law_id] = law_ids.get(law_id, 0) + 1
        
        if len(law_ids) > 1:
            print(f"\n📊 Breakdown by law_id:")
            for law_id, count in sorted(law_ids.items(), key=lambda x: x[1], reverse=True):
                print(f"   - {law_id}: {count} chunks")
        
        print(f"\n💡 Next step:")
        next_category = category or 'YOUR_CATEGORY'
        print(f"   python embed_and_upload.py --chunk-file \"{output_path}\" --category {next_category}")


if __name__ == "__main__":
    main()

