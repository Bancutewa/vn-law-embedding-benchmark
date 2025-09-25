#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(__file__))

from modules.chunking import generate_law_id

# Test với file name thực tế
file_name = 'luat_hon_nhan_gia_dinh_2014.docx'
result = generate_law_id(file_name)
print(f'File: {file_name}')
print(f'Law ID: {result}')
