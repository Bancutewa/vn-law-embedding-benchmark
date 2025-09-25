#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Loading Module
Load và xử lý documents và benchmark queries
"""

import json
import os
import sys
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

from .chunking import read_docx, chunk_law_document, generate_law_id

def load_all_law_documents() -> List[Dict[str, Any]]:
    """Load và chunk tất cả văn bản luật từ thư mục law_content"""
    print("📚 Loading ALL law documents from law_content folder...")

    # Load danh sách file từ JSON
    law_file_paths = []
    try:
        with open("data_files/law_file_paths.json", 'r', encoding='utf-8') as f:
            law_file_paths = json.load(f)
        print(f"✅ Loaded {len(law_file_paths)} law file paths from JSON")
    except FileNotFoundError:
        print("❌ data_files/law_file_paths.json not found! Please run find_law_files.py first.")
        print("🔄 Running find_law_files.py to generate the file...")
        try:
            import subprocess
            result = subprocess.run([sys.executable, "find_law_files.py"],
                                  capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode == 0:
                print("✅ Successfully generated data_files/law_file_paths.json")
                with open("data_files/law_file_paths.json", 'r', encoding='utf-8') as f:
                    law_file_paths = json.load(f)
                print(f"✅ Loaded {len(law_file_paths)} law file paths from generated JSON")
            else:
                print(f"❌ Failed to generate data_files/law_file_paths.json: {result.stderr}")
                return []
        except Exception as e:
            print(f"❌ Error running find_law_files.py: {e}")
            return []
    except Exception as e:
        print(f"❌ Error loading data_files/law_file_paths.json: {e}")
        return []

    all_law_docs = []
    successful_files = 0
    failed_files = 0

    print(f"\n🔄 Processing {len(law_file_paths)} law files...")

    for i, file_info in enumerate(tqdm(law_file_paths, desc="Processing law files")):
        file_path = file_info['path']
        category = file_info['category']
        file_name = file_info['file_name']

        print(f"\n📄 [{i+1}/{len(law_file_paths)}] Processing: {file_name}")
        print(f"   Category: {category}")
        print(f"   Path: {file_path}")

        if not os.path.exists(file_path):
            print(f"   ❌ File not found: {file_path}")
            failed_files += 1
            continue

        try:
            # Bước 1: Đọc file
            print(f"   📖 Reading file...")
            law_text = read_docx(file_path)  # Hàm này đã xử lý cả .doc và .docx

            if not law_text or len(law_text.strip()) < 100:
                print(f"   ⚠️ File seems empty or too short: {len(law_text)} characters")
                print(f"   ⏭️ Skipping file (cannot read content)")
                failed_files += 1
                continue

            # Bước 2: Chia thành chunks
            print(f"   🔨 Chunking document...")
            # Tạo law_id tự động từ tên file
            law_id = generate_law_id(file_name)
            print(f"   📋 Generated law_id: {law_id}")
            law_chunks = chunk_law_document(law_text, law_id=law_id, law_no="", law_title=file_name)

            if not law_chunks:
                print(f"   ⚠️ No chunks created from file")
                failed_files += 1
                continue

            # Bước 3: Chuẩn bị dữ liệu cho đánh giá
            print(f"   🗂️ Preparing chunks...")
            for j, chunk in enumerate(law_chunks):
                # Thêm thông tin về file gốc vào metadata
                chunk_metadata = chunk.get('metadata', {}).copy()
                chunk_metadata.update({
                    'source_file': file_path,
                    'source_category': category,
                    'source_file_name': file_name,
                    'chunk_index': j
                })

                # Tạo document theo format hn2014_chunks.json
                all_law_docs.append({
                    'id': chunk['id'],
                    'content': chunk['content'],
                    'metadata': chunk_metadata
                })

            print(f"   ✅ Successfully processed {len(law_chunks)} chunks")
            successful_files += 1

        except Exception as e:
            print(f"   ❌ Error processing file: {e}")
            failed_files += 1
            continue

    print(f"\n📊 Processing Summary:")
    print(f"   ✅ Successfully processed: {successful_files} files")
    print(f"   ❌ Failed to process: {failed_files} files")
    print(f"   📄 Total chunks created: {len(all_law_docs)}")

    if all_law_docs:
        import numpy as np
        print(f"   📊 Average chunk length: {np.mean([len(doc['content']) for doc in all_law_docs]):.0f} characters")

        # Thống kê theo category
        category_counts = {}
        for doc in all_law_docs:
            doc_category = doc.get('metadata', {}).get('source_category', 'Unknown')
            category_counts[doc_category] = category_counts.get(doc_category, 0) + 1

        print(f"\n📈 Chunk distribution by category:")
        for category, count in category_counts.items():
            print(f"   - {category}: {count} chunks")

    print(f"\n✅ Successfully loaded {len(all_law_docs)} law document chunks from {successful_files} files!")
    return all_law_docs

def load_question_benchmark(random_sample: int = 50) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Load các câu hỏi từ file Excel để làm benchmark queries"""
    try:
        with open("data_files/law_questions.json", 'r', encoding='utf-8') as f:
            questions = json.load(f)

        print(f"✅ Loaded {len(questions)} benchmark questions from Excel files")

        # Random sample 50 questions
        if len(questions) > random_sample:
            import random
            questions_sample = random.sample(questions, random_sample)
            print(f"🎲 Randomly selected {random_sample} questions for evaluation")
        else:
            questions_sample = questions
            print(f"📊 Using all {len(questions)} questions (less than {random_sample})")

        # Trả về list các query (chỉ lấy primary_query)
        queries = [q['primary_query'] for q in questions_sample]

        # Thống kê theo category
        category_stats = {}
        for q in questions_sample:
            cat = q.get('full_category', 'Unknown')
            category_stats[cat] = category_stats.get(cat, 0) + 1

        print(f"📊 Sample distribution by category:")
        for cat, count in sorted(category_stats.items()):
            print(f"   - {cat}: {count} questions")

        return queries, questions_sample

    except FileNotFoundError:
        print("⚠️ data_files/law_questions.json not found. Using default queries.")
        return get_default_benchmark_queries(), []
    except Exception as e:
        print(f"⚠️ Error loading questions: {e}. Using default queries.")
        return get_default_benchmark_queries(), []

def get_default_benchmark_queries() -> List[str]:
    """Các query mặc định nếu không có file questions"""
    return [
        "Người bị bệnh tâm thần là người mất năng lực hành vi dân sự là đúng hay sai?",
        "Sự thỏa thuận của các bên không vi phạm điều cấm của pháp luật, không trái đạo đức xã hội thì được gọi là hợp đồng là đúng hay sai?",
        "Tiền phúng viếng đám ma cũng thuộc di sản thừa kế của người chết để lại là đúng hay sai?",
        "Cá nhân được hưởng di sản thừa kế phải trả nợ thay cho người để lại di sản là đúng hay sai?",
        "Hợp đồng vô hiệu là hợp đồng vi phạm pháp luật là đúng hay sai?",
        "Hoa lợi, lợi tức phát sinh từ tài sản riêng của một bên vợ hoặc chồng sẽ là tài sản chung nếu hoa lợi, lợi tức đó là nguồn sống duy nhất của gia đình.",
        "Hội Liên hiệp phụ nữ có quyền yêu cầu Tòa án ra quyết định hủy kết hôn trái pháp luật do vi phạm sự tự nguyện.",
        "Hôn nhân chỉ chấm dứt khi một bên vợ, chồng chết.",
        "Kết hôn có yếu tố nước ngoài có thể đăng ký tại UBND cấp xã.",
        "Khi cha mẹ không thể nuôi dưỡng, cấp dưỡng được cho con, thì ông bà phải có nghĩa vụ nuôi dưỡng hoặc cấp dưỡng cho cháu.",
        "Khi hôn nhân chấm dứt, mọi quyền và nghĩa vụ giữa những người đã từng là vợ chồng cũng chấm dứt.",
        "Khi vợ chồng ly hôn, con dưới 36 tháng tuổi được giao cho người vợ trực tiếp nuôi dưỡng.",
        "Khi vợ hoặc chồng thực hiện những giao dịch phục vụ cho nhu cầu thiết yếu của gia đình mà không có sự đồng ý của bên kia thì người thực hiện giao dịch đó phải thanh toán bằng tài sản riêng của mình.",
        "Khi không sống chung cùng với cha mẹ, con đã thành niên có khả năng lao động phải cấp dưỡng cho cha mẹ.",
        "Chỉ UBND cấp tỉnh nơi công dân Việt Nam cư trú mới có thẩm quyền đăng ký việc kết hôn giữa công dân Việt Nam với người nước ngoài.",
        "Điều kiện kết hôn của nam và nữ theo pháp luật Việt Nam là gì?",
        "Tài sản nào được coi là tài sản chung của vợ chồng?",
        "Thủ tục ly hôn tại tòa án được quy định như thế nào?",
        "Quyền và nghĩa vụ của cha mẹ đối với con chưa thành niên?",
        "Trường hợp nào vợ chồng có thể thỏa thuận về chế độ tài sản?"
    ]

def save_chunks_to_json(chunks: List[Dict[str, Any]], output_path: str) -> None:
    """Lưu chunks vào file JSON"""
    safe_path = output_path.encode('ascii', 'ignore').decode('ascii') or output_path
    print(f"Saving {len(chunks)} chunks to {safe_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    safe_path = output_path.encode('ascii', 'ignore').decode('ascii') or output_path
    print(f"Saved {len(chunks)} chunks to {safe_path}")

def load_chunks_from_json(input_path: str) -> List[Dict[str, Any]]:
    """Load chunks từ file JSON"""
    safe_path = input_path.encode('ascii', 'ignore').decode('ascii') or input_path
    print(f"Loading chunks from {safe_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    safe_path = input_path.encode('ascii', 'ignore').decode('ascii') or input_path
    print(f"Loaded {len(chunks)} chunks from {safe_path}")
    return chunks
