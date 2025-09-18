#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test để xem file nào có thể đọc được
"""

import os
import json
from docx import Document
import docx2txt

def read_docx_safe(file_path):
    """Đọc file docx một cách an toàn"""
    try:
        # Thử đọc bằng python-docx trước
        doc = Document(file_path)
        text = "\n".join((p.text or "").strip() for p in doc.paragraphs)
        if text and len(text.strip()) > 0:
            return text, "python-docx"
        else:
            return "", "python-docx (empty)"
    except Exception as e1:
        # Thử dùng docx2txt
        try:
            text = docx2txt.process(file_path)
            if text and len(text.strip()) > 0:
                return text, "docx2txt"
            else:
                return "", "docx2txt (empty)"
        except Exception as e2:
            return "", f"failed: {e1}"

def test_all_files():
    """Test tất cả file"""
    
    print("🧪 Testing all law files for readability...")
    
    # Load danh sách file từ JSON
    try:
        with open("law_file_paths.json", 'r', encoding='utf-8') as f:
            law_file_paths = json.load(f)
        print(f"✅ Loaded {len(law_file_paths)} law file paths from JSON")
    except Exception as e:
        print(f"❌ Error loading law_file_paths.json: {e}")
        return
    
    readable_files = []
    unreadable_files = []
    
    for i, file_info in enumerate(law_file_paths):
        file_path = file_info['path']
        file_name = file_info['file_name']
        
        print(f"\n📄 [{i+1}/{len(law_file_paths)}] Testing: {file_name}")
        
        if not os.path.exists(file_path):
            print(f"   ❌ File not found")
            unreadable_files.append(file_info)
            continue
        
        text, method = read_docx_safe(file_path)
        
        if text:
            print(f"   ✅ Successfully read {len(text):,} characters using {method}")
            readable_files.append(file_info)
        else:
            print(f"   ❌ Cannot read file: {method}")
            unreadable_files.append(file_info)
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Readable files: {len(readable_files)}")
    print(f"   ❌ Unreadable files: {len(unreadable_files)}")
    
    if readable_files:
        print(f"\n📋 Readable files:")
        for file_info in readable_files:
            print(f"   - {file_info['file_name']} ({file_info['category']})")
    
    if unreadable_files:
        print(f"\n📋 Unreadable files:")
        for file_info in unreadable_files:
            print(f"   - {file_info['file_name']} ({file_info['category']})")

if __name__ == "__main__":
    test_all_files()
