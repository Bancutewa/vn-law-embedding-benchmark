#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để tìm và đọc tất cả file câu hỏi Excel (.xlsx) trong thư mục law_content
"""

import os
import glob
import json
from pathlib import Path

def read_excel_questions(file_path):
    """Đọc file Excel chứa câu hỏi và trả về danh sách câu hỏi"""
    try:
        import pandas as pd

        print(f"   📖 Reading Excel file: {os.path.basename(file_path)}")

        # Đọc file Excel
        df = pd.read_excel(file_path)

        print(f"   📊 Found {len(df)} rows, columns: {list(df.columns)}")

        questions = []

        # Chuẩn hóa tên cột (loại bỏ khoảng trắng thừa)
        df.columns = df.columns.str.strip()

        # Tìm các cột có thể chứa query, positive, negative
        query_col = None
        positive_col = None
        negative_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'query' in col_lower or 'câu hỏi' in col_lower or 'question' in col_lower:
                query_col = col
            elif 'positive' in col_lower or 'tích cực' in col_lower or 'đúng' in col_lower:
                positive_col = col
            elif 'negative' in col_lower or 'tiêu cực' in col_lower or 'sai' in col_lower:
                negative_col = col

        print(f"   🔍 Detected columns - Query: {query_col}, Positive: {positive_col}, Negative: {negative_col}")

        # Đọc từng dòng
        for idx, row in df.iterrows():
            question_data = {
                'id': f"{os.path.basename(file_path).replace('.xlsx', '')}_Q{idx+1}",
                'query': str(row[query_col]).strip() if query_col and pd.notna(row[query_col]) else "",
                'positive': str(row[positive_col]).strip() if positive_col and pd.notna(row[positive_col]) else "",
                'negative': str(row[negative_col]).strip() if negative_col and pd.notna(row[negative_col]) else "",
                'source_file': os.path.basename(file_path),
                'row_index': idx + 1
            }

            # Chỉ thêm nếu có query
            if question_data['query'] and question_data['query'] != 'nan':
                questions.append(question_data)
            else:
                print(f"   ⚠️ Skipping row {idx+1}: empty query")

        print(f"   ✅ Extracted {len(questions)} valid questions")
        return questions

    except ImportError:
        print("   ❌ pandas not installed. Install with: pip install pandas openpyxl")
        return []
    except Exception as e:
        print(f"   ❌ Error reading Excel file: {e}")
        return []

def find_all_question_files(law_content_dir="law_content"):
    """Tìm tất cả file câu hỏi Excel (.xlsx) trong thư mục law_content"""

    print(f"🔍 Searching for question Excel files in: {law_content_dir}")

    # Tìm tất cả file .xlsx
    xlsx_files = glob.glob(os.path.join(law_content_dir, "**", "*.xlsx"), recursive=True)

    # Lọc bỏ các file tạm thời (bắt đầu bằng ~$)
    xlsx_files = [f for f in xlsx_files if not os.path.basename(f).startswith('~$')]

    print(f"📊 Found {len(xlsx_files)} Excel files:")

    # Phân loại theo thư mục
    question_files_by_category = {}

    for file_path in xlsx_files:
        # Lấy đường dẫn tương đối từ law_content
        rel_path = os.path.relpath(file_path, law_content_dir)

        # Phân tích cấu trúc thư mục
        path_parts = rel_path.split(os.sep)

        if len(path_parts) >= 2:
            main_category = path_parts[0]  # Ví dụ: "Bất động sản"
            sub_category = path_parts[1]  # Ví dụ: "Luật Đất Đai"

            if main_category not in question_files_by_category:
                question_files_by_category[main_category] = {}
            if sub_category not in question_files_by_category[main_category]:
                question_files_by_category[main_category][sub_category] = []

            question_files_by_category[main_category][sub_category].append({
                'file_path': file_path,
                'relative_path': rel_path,
                'file_name': os.path.basename(file_path)
            })

    # In kết quả phân loại
    print(f"\n📁 Question files by category:")
    for main_cat, sub_cats in question_files_by_category.items():
        print(f"\n🏛️ {main_cat}:")
        for sub_cat, files in sub_cats.items():
            print(f"   📂 {sub_cat}: {len(files)} files")
            for file_info in files:
                print(f"      - {file_info['relative_path']}")

    return xlsx_files, question_files_by_category

def process_all_question_files(question_files_by_category):
    """Xử lý tất cả file câu hỏi và tạo danh sách câu hỏi"""

    print(f"\n📖 Processing all question files...")

    all_questions = []

    for main_cat, sub_cats in question_files_by_category.items():
        for sub_cat, files in sub_cats.items():
            print(f"\n🏛️ Processing category: {main_cat} > {sub_cat}")

            for file_info in files:
                file_path = file_info['file_path']

                print(f"   📄 Processing: {file_info['relative_path']}")

                # Đọc câu hỏi từ file
                questions = read_excel_questions(file_path)

                # Thêm thông tin category vào mỗi câu hỏi
                for question in questions:
                    question.update({
                        'main_category': main_cat,
                        'sub_category': sub_cat,
                        'full_category': f"{main_cat} - {sub_cat}"
                    })

                all_questions.extend(questions)

                print(f"   📊 Total questions so far: {len(all_questions)}")

    return all_questions

def save_questions_to_json(questions, output_file="data_files/law_questions.json"):
    """Lưu danh sách câu hỏi vào file JSON"""

    print(f"\n💾 Saving {len(questions)} questions to: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(questions)} questions to {output_file}")

def create_query_variations(questions):
    """Tạo các biến thể của query để tăng đa dạng"""

    print(f"\n🔄 Creating query variations...")

    enhanced_questions = []

    for question in questions:
        base_query = question['query']

        # Tạo các biến thể
        variations = [
            base_query,  # Query gốc
        ]

        # Thêm biến thể với dấu chấm hỏi nếu chưa có
        if not base_query.endswith('?'):
            variations.append(base_query + '?')

        # Thêm biến thể viết thường
        if base_query != base_query.lower():
            variations.append(base_query.lower())

        # Lưu query variations
        question['query_variations'] = list(set(variations))  # Loại bỏ trùng lặp
        question['primary_query'] = variations[0]

        enhanced_questions.append(question)

    print(f"✅ Created variations for {len(enhanced_questions)} questions")
    return enhanced_questions

if __name__ == "__main__":
    # Tìm tất cả file câu hỏi Excel
    xlsx_files, question_files_by_category = find_all_question_files()

    if not xlsx_files:
        print("❌ No Excel question files found!")
        exit(1)

    # Xử lý tất cả file
    all_questions = process_all_question_files(question_files_by_category)

    # Tạo biến thể query
    enhanced_questions = create_query_variations(all_questions)

    # Lưu vào JSON
    save_questions_to_json(enhanced_questions)

    # Thống kê
    print(f"\n📊 SUMMARY:")
    print(f"   📁 Categories processed: {len(question_files_by_category)}")
    print(f"   📄 Excel files processed: {len(xlsx_files)}")
    print(f"   ❓ Total questions extracted: {len(enhanced_questions)}")

    if enhanced_questions:
        # Thống kê theo category
        category_stats = {}
        for q in enhanced_questions:
            cat = q.get('full_category', 'Unknown')
            category_stats[cat] = category_stats.get(cat, 0) + 1

        print(f"   📈 Questions by category:")
        for cat, count in sorted(category_stats.items()):
            print(f"      - {cat}: {count} questions")

        # Hiển thị mẫu
        print(f"\n📝 SAMPLE QUESTION:")
        sample = enhanced_questions[0]
        print(f"   ID: {sample['id']}")
        print(f"   Query: {sample['query']}")
        print(f"   Positive: {sample['positive'][:100]}..." if sample['positive'] else "   Positive: (empty)")
        print(f"   Negative: {sample['negative'][:100]}..." if sample['negative'] else "   Negative: (empty)")
        print(f"   Category: {sample['full_category']}")

    print(f"\n🎉 Completed! Question extraction finished.")
