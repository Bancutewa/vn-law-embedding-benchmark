#!/usr/bin/env python3

import sys
import os
sys.path.append('modules')

from chunking import generate_law_id

# Test với file name thực tế
file_name = 'luat_hon_nhan_gia_dinh_2014.docx'
result = generate_law_id(file_name)
print(f'File: {file_name}')
print(f'Law ID: {result}')

# Test từng bước
name = file_name.replace('.docx', '').replace('.doc', '').strip()
name_normalized = name.lower().replace('_', ' ')
name_lower = name.lower()

print(f"name: '{name}'")
print(f"name_normalized: '{name_normalized}'")
print(f"name_lower: '{name_lower}'")

# Test từng key
law_mappings = {
    'kinh doanh bất động sản': 'LKBDS',
    'nhà ở': 'LNHAO',
    'đất đai': 'LDATDAI',
    'đầu tư': 'LDAUTU',
    'đầu tư công': 'LDAUTUCONG',
    'đầu tư theo phương thức đối tác công tư': 'LDAUTUPPPCT',
    'thuế sử dụng đất nông nghiệp': 'LTSDDNONGNGHIEP',
    'thuế sử dụng đất phi nông nghiệp': 'LTSDDPHINONGNGHIEP',
    'xây dựng': 'LXAYDUNG',
    'hôn nhân và gia đình': 'LHNVDG',
    'hon nhan va gia dinh': 'LHNVDG',
    'luat_hon_nhan_gia_dinh_2014': 'LHNVDG',
}

for key, value in law_mappings.items():
    import unicodedata
    key_normalized = unicodedata.normalize('NFD', key).encode('ascii', 'ignore').decode('ascii').lower()

    # Check exact match
    exact_match = key in name_normalized or key in name_lower or key_normalized in name_normalized
    if exact_match:
        print(f"EXACT MATCH: key='{key}', value='{value}'")
        break

    # Check word match
    key_words = key.split()
    if len(key_words) >= 2:
        key_words_normalized = [unicodedata.normalize('NFD', w).encode('ascii', 'ignore').decode('ascii').lower()
                               for w in key_words]
        word_match = all(word in name_normalized for word in key_words) or \
                    all(word in name_normalized for word in key_words_normalized)
        if word_match:
            print(f"WORD MATCH: key='{key}', value='{value}'")
            break
