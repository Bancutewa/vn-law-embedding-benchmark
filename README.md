# 🔍 Vietnamese Legal Embedding Model Evaluation & Testing

Dự án này đánh giá và so sánh các mô hình embedding cho văn bản luật tiếng Việt, với giao diện Gradio để test interactive.

## 📋 Tổng quan

- **Dataset:** Luật Việt Nam từ nhiều lĩnh vực (hiện tại: 511 chunks từ 7 luật)
- **Benchmark Questions:** 511 câu hỏi từ Excel files → Random 50 câu cho evaluation
- **Models:** 4 mô hình embedding Vietnamese và multilingual
- **Chunking:** 2-pass system với clause intro injection & law_id tự động
- **Storage:** Qdrant vector database với batch upsert
- **Analysis:** **Top 3 queries** với **full content** và thông tin đối chiếu
- **Output:** `embedding_evaluation_results.json` với detailed analysis

## 📁 Cấu trúc thư mục

```
vn-law-embedding-benchmark/
├── embedding_evaluation.py       # Main evaluation script
├── find_law_files.py            # Law document discovery
├── find_question_files.py       # Question extraction from Excel
├── README.md                    # Documentation
├── requirements.txt             # Dependencies
├── setup_environment.bat        # Environment setup
├── Danh_Gia_Mo_Hinh_Embedding.ipynb  # Jupyter notebook
├── data_files/                  # Data files
│   ├── law_file_paths.json      # Law document metadata
│   └── law_questions.json       # Benchmark questions
├── results/                     # Output files
│   ├── embedding_evaluation_results.json
│   └── top_queries_analysis.json
├── utilities/                   # Utility scripts
│   ├── view_top_queries.py      # View top queries results
│   ├── test_chunking.py         # Chunking tests
│   └── test_readable_files.py   # File reading tests
└── law_content/                 # Raw law documents
```

## 🚀 Cách sử dụng

### 1. Chuẩn bị dữ liệu

```bash
# 1. Tìm và catalog tất cả file luật
python find_law_files.py

# 2. Trích xuất câu hỏi từ file Excel
python find_question_files.py

# Output files:
# - data_files/law_file_paths.json: Danh sách file luật
# - data_files/law_questions.json: 511 câu hỏi benchmark
```

### 2. Chạy đánh giá với 50 câu hỏi random

```bash
# Chạy full evaluation với 50 queries random + hiển thị top 3
python embedding_evaluation.py

# Output:
# - results/embedding_evaluation_results.json: Kết quả đánh giá đầy đủ (ghi đè)
# - results/top_queries_analysis.json: Top queries analysis sạch (tự động tạo)
# - Console: Top 3 queries với full content và thông tin đối chiếu


## 📊 Output Format

### Top 3 Queries Display
```

🔝 TOP 3 QUERIES BY TOTAL SCORE (Best Model)

1.  Total Score: 12.456
    Query: [Câu hỏi từ Excel]
    Category: Bất động sản - Luật Đất Đai
    Expected Answer: [Câu trả lời tích cực từ Excel]...
    Negative Answer: [Câu trả lời tiêu cực từ Excel]...

    📄 Rank 1: Score 0.823 | Law: LDATDAI
    📝 Citation: Điều 1
    📖 Content: Điều 1. Phạm vi điều chỉnh Luật này quy định...
    ... (500 chars total)

    📄 Rank 2: Score 0.756 | Law: LDATDAI
    📝 Citation: Điều 2
    📖 Content: Điều 2. Đối tượng áp dụng...
    ... (300 chars total)

````

### JSON Results
```json
{
  "model_name": "minhquan6203/paraphrase-vietnamese-law",
  "top_queries_by_total_score": [
    {
      "rank": 1,
      "query": "Câu hỏi...",
      "total_score": 12.456,
      "question_info": {
        "category": "Bất động sản - Luật Đất Đai",
        "positive_answer": "Câu trả lời tích cực...",
        "negative_answer": "Câu trả lời tiêu cực...",
        "source_file": "Luật đất đai 2024_.xlsx"
      },
      "top_answers": [
        {
          "rank": 1,
          "score": 0.823,
          "citation": "Điều 1",
          "law_id": "LDATDAI",
          "content": "Full content...",
          "metadata": {...}
        }
      ]
    }
  ]
}
````

### 3. Cài đặt môi trường

```bash
# Tạo virtual environment
python -m venv legal_ai_env
source legal_ai_env/bin/activate  # Linux/Mac

# CLI
setup_environment.bat # Windows

# hoặc
legal_ai_env\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

Script này sẽ:

- ✅ Load và chunk dữ liệu luật từ DOCX (tự động tạo law_id)
- ✅ Trích xuất câu hỏi từ Excel files (format Query-Positive-Negative)
- ✅ **Random 50 câu hỏi** từ 511 câu benchmark để đánh giá
- ✅ Đánh giá 4 mô hình embedding với 50 câu hỏi đã chọn
- ✅ Lưu trữ embeddings trong Qdrant vector database
- ✅ **Hiển thị TOP 3 câu hỏi** có tổng score cao nhất với **full content**
- ✅ Tạo báo cáo chi tiết với ranking và phân tích
- ✅ Ghi đè file `embedding_evaluation_results.json`

## 🤖 Models được đánh giá

1. **🏆 minhquan6203/paraphrase-vietnamese-law** (Best)
2. **sentence-transformers/paraphrase-multilingual-mpnet-base-v2**
3. **namnguyenba2003/Vietnamese_Law_Embedding_finetuned_v3_256dims**
4. **truro7/vn-law-embedding**

## 📊 Dataset & Questions

### Luật Documents

- **Tự động phát hiện:** Script `find_law_files.py` tìm tất cả file .docx trong `law_content/`
- **Chunking thông minh:** Chia theo cấu trúc Điều-Khoản-Điểm với law_id tự động
- **Law ID mapping:**
  - `LKBDS` → Luật Kinh Doanh Bất Động Sản
  - `LNHAO` → Luật Nhà Ở
  - `LDATDAI` → Luật Đất Đai
  - `LDAUTU` → Luật Đầu Tư
  - `LTSDDNONGNGHIEP` → Luật Thuế Sử Dụng Đất Nông Nghiệp
  - `LTSDDPHINONGNGHIEP` → Luật Thuế Sử Dụng Đất Phi Nông Nghiệp
  - `LXAYDUNG` → Luật Xây Dựng

### Benchmark Questions

- **Nguồn:** File Excel .xlsx trong thư mục `Câu hỏi/`
- **Format:** Mỗi file có 3 cột: `Query`, `Positive`, `Negative`
- **Tự động trích xuất:** Script `find_question_files.py` xử lý tất cả file Excel
- **Số lượng:** 511 câu hỏi từ 4 luật bất động sản

```bash
# Trích xuất câu hỏi từ Excel
python find_question_files.py

# Output: law_questions.json với cấu trúc
{
  "id": "file_name_Q1",
  "query": "Câu hỏi...",
  "positive": "Câu trả lời tích cực...",
  "negative": "Câu trả lời tiêu cực...",
  "query_variations": ["query1", "query2", ...],
  "full_category": "Bất động sản - Luật Đất Đai"
}
```

## 🎯 Interface Features

### Search & Testing

- Real-time semantic search
- Top-K adjustable (5-20 results)
- Detailed metrics display
- Error handling robust

### Results Display

- Similarity scores
- Document metadata
- Content preview
- Performance metrics

## 📊 Evaluation Metrics

- **Max Score:** Điểm similarity cao nhất
- **Average Top-5:** Trung bình top-5 results
- **Above 0.7/0.5:** Số results trên ngưỡng
- **Cosine Similarity:** Phương pháp so sánh

## 📁 Cấu trúc files

```
vn-law-embedding-benchmark/
├── Danh_Gia_Mo_Hinh_Embedding.ipynb    # Main evaluation script
├── requirements.txt                  # Dependencies
├── setup_environment.bat            # Windows setup script
├── README.md                        # Documentation
└── LuatHonNhan/                     # Data folder
    └── luat_hon_nhan_va_gia_dinh.docx
```

## 🔧 System Requirements

- **Python:** 3.8+
- **RAM:** 8GB+ (cho model loading)
- **GPU:** Optional (CUDA support)
- **Storage:** 2GB+ cho models

## 📝 Example Queries

```
"Điều kiện kết hôn của nam và nữ theo pháp luật Việt Nam là gì?"
"Tài sản nào được coi là tài sản chung của vợ chồng?"
"Thủ tục ly hôn tại tòa án được quy định như thế nào?"
"Quyền và nghĩa vụ của cha mẹ đối với con chưa thành niên?"
```

## 🎉 Results Summary

**Best Model:** `minhquan6203/paraphrase-vietnamese-law`

- **Score:** 0.9034
- **Type:** Transformers
- **Specialization:** Fine-tuned cho luật Việt Nam
