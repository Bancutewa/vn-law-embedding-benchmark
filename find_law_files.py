#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để tìm tất cả file văn bản pháp lý trong thư mục law_content
"""

import os
import glob
from pathlib import Path

def find_all_law_files(law_content_dir="law_content"):
    """Tìm tất cả file văn bản pháp lý trong thư mục law_content"""
    
    print(f"🔍 Searching for law files in: {law_content_dir}")
    
    # Tìm tất cả file .doc và .docx
    doc_files = glob.glob(os.path.join(law_content_dir, "**", "*.doc"), recursive=True)
    docx_files = glob.glob(os.path.join(law_content_dir, "**", "*.docx"), recursive=True)
    
    all_files = doc_files + docx_files
    
    # Lọc chỉ các file trong thư mục có tên chứa "Luật_" hoặc "văn bản pháp luật"
    filtered_files = []
    for file_path in all_files:
        # Lấy đường dẫn tương đối từ law_content
        rel_path = os.path.relpath(file_path, law_content_dir)
        path_parts = rel_path.split(os.sep)
        
        # Kiểm tra xem có thư mục nào trong đường dẫn chứa "Luật_" hoặc "văn bản pháp luật" không
        is_law_folder = False
        for part in path_parts:
            if "Luật_" in part or "văn bản pháp luật" in part.lower() or "Văn bản pháp lý" in part or  "văn bản quy phạm pháp luật" in part.lower():
                is_law_folder = True
                break
        
        if is_law_folder:
            filtered_files.append(file_path)
        else:
            print(f"   ⏭️ Skipping file (not in law folder): {rel_path}")
    
    all_files = filtered_files
    
    print(f"📊 Found {len(all_files)} law files (filtered by law folders):")
    print(f"   - Total .doc files found: {len(doc_files)}")
    print(f"   - Total .docx files found: {len(docx_files)}")
    print(f"   - Files in law folders: {len(all_files)}")
    
    # Phân loại theo thư mục
    law_files_by_category = {}
    
    for file_path in all_files:
        # Lấy đường dẫn tương đối từ law_content
        rel_path = os.path.relpath(file_path, law_content_dir)
        
        # Phân tích cấu trúc thư mục
        path_parts = rel_path.split(os.sep)
        
        if len(path_parts) >= 2:
            main_category = path_parts[0]  # Ví dụ: "Bất động sản"
            sub_category = path_parts[1]  # Ví dụ: "Luật Đất Đai"
            
            if main_category not in law_files_by_category:
                law_files_by_category[main_category] = {}
            if sub_category not in law_files_by_category[main_category]:
                law_files_by_category[main_category][sub_category] = []
            
            # Lấy extension chính xác
            file_name = os.path.basename(file_path)
            if '.' in file_name:
                file_extension = '.' + file_name.split('.')[-1]
            else:
                file_extension = ''
            
            law_files_by_category[main_category][sub_category].append({
                'file_path': file_path,
                'relative_path': rel_path,
                'file_name': file_name,
                'file_extension': file_extension
            })
    
    # In kết quả phân loại
    print(f"\n📁 Law files by category:")
    for main_cat, sub_cats in law_files_by_category.items():
        print(f"\n🏛️ {main_cat}:")
        for sub_cat, files in sub_cats.items():
            print(f"   📂 {sub_cat}: {len(files)} files")
            for file_info in files:
                print(f"      - {file_info['relative_path']}")
    
    return all_files, law_files_by_category

def create_law_file_paths_list(law_files_by_category):
    """Tạo danh sách các đường dẫn file để sử dụng trong load_law_documents"""
    
    print(f"\n📝 Creating law file paths list...")
    
    law_file_paths = []
    
    for main_cat, sub_cats in law_files_by_category.items():
        for sub_cat, files in sub_cats.items():
            for file_info in files:
                law_file_paths.append({
                    'path': file_info['file_path'],
                    'relative_path': file_info['relative_path'],
                    'category': f"{main_cat} - {sub_cat}",
                    'file_name': file_info['file_name'],
                    'extension': file_info['file_extension']
                })
    
    print(f"✅ Created {len(law_file_paths)} law file paths")
    
    return law_file_paths

def save_law_file_paths(law_file_paths, output_file="data_files/law_file_paths.json"):
    """Lưu danh sách đường dẫn file vào file JSON"""
    
    import json
    
    print(f"\n💾 Saving law file paths to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(law_file_paths, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Saved {len(law_file_paths)} file paths to {output_file}")

if __name__ == "__main__":
    # Tìm tất cả file văn bản pháp lý
    all_files, law_files_by_category = find_all_law_files()
    
    # Tạo danh sách đường dẫn
    law_file_paths = create_law_file_paths_list(law_files_by_category)
    
    # Lưu vào file JSON
    save_law_file_paths(law_file_paths)
    
    print(f"\n🎉 Completed! Found {len(all_files)} law files total.")
