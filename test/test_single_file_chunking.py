#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script test để demo chunking file đơn lẻ
"""

import os
import sys

# Test chunking file đơn lẻ
def test_single_file_chunking():
    file_path = "law_content\Quyền dân sự_\Luật nuôi con nuôi\Luật_\Luật nuôi con nuôi.docx"

    if os.path.exists(file_path):
        print(f"Testing single file chunking: {file_path}")
        cmd = f'python chunking.py --file "{file_path}"'
        print(f"Command: {cmd}")

        # Thực thi command
        os.system(cmd)
    else:
        print(f"File not found: {file_path}")
        print("Available .docx files in Quyền dân sự_:")

        # Tìm các file .docx có thể đọc
        import glob
        docx_files = glob.glob("law_content/Quyền dân sự_/**/*.docx", recursive=True)
        for f in docx_files[:5]:  # Hiển thị 5 file đầu
            print(f"  - {f}")

if __name__ == "__main__":
    test_single_file_chunking()
