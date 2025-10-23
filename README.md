# 🔍 Vietnamese Legal Embedding Benchmark

Dự án đánh giá và so sánh các mô hình embedding cho văn bản luật tiếng Việt

### **🔬 BENCHMARK EVALUATION (Đánh giá Benchmark)**

**Script chính:** `main_refactored.py` - Hệ thống đánh giá benchmark hoàn chỉnh

- ✅ **Đánh giá nhiều model** cùng lúc
- ✅ **So sánh performance** chi tiết
- ✅ **Tạo báo cáo ranking** và recommendations
- ✅ **Top queries analysis** với full content
- ✅ **Workflow hoàn chỉnh**: Chunk → Embed → Evaluate → Report

### **📤 CHUNKING & UPLOAD (Chunk và Tải lên)**

**Workflow đơn giản 3 bước:**

1. **`chunk_and_save.py`** - Chunk 1 hoặc nhiều files

   - Config sẵn `DEFAULT_FILES` trong code
   - Output: `data/{CATEGORY}/*.json`

2. **`merge_chunks.py`** - Merge tất cả chunks

   - Config sẵn `DEFAULT_DIRECTORY` trong code
   - Output: `data/{CATEGORY}_merged_*.json`

3. **`embed_and_upload.py`** - Upload lên Qdrant
   - Config sẵn `--chunk-file` trong code
   - Output: Qdrant collection

## 📋 Tổng quan Dataset & Models

- **Dataset:** Luật Việt Nam từ nhiều lĩnh vực (Bất động sản, Doanh nghiệp, Thương mại, Quyền dân sự)
- **Benchmark Questions:** 511 câu hỏi từ Excel files → Random 50 câu cho evaluation
- **Models:** 4 mô hình embedding Vietnamese và multilingual
- **Chunking:** 2-pass system với clause intro injection & law_id tự động
- **Storage:** Qdrant vector database với batch upsert
- **Analysis:** **Top 3 queries** với **full content** và thông tin đối chiếu

## 🚀 Hướng dẫn Chi tiết

### **Bước 1: Chuẩn bị dữ liệu chung**

```bash
# 1. Tìm và catalog tất cả file luật theo category
python find_law_files.py

# 2. Trích xuất câu hỏi từ file Excel
python find_question_files.py

# Output files:
# - data_files/law_file_paths.json: Danh sách tất cả file luật
# - data_files/law_questions.json:  Câu hỏi benchmark
# - data_files/BDS/bds_file_paths.json: Files Bất động sản
# - data_files/DN/dn_file_paths.json: Files Doanh nghiệp
# - data_files/TM/tm_file_paths.json: Files Thương mại
# - data_files/QDS/qds_file_paths.json: Files Quyền dân sự
```

---

## 🔬 **BENCHMARK EVALUATION (main_refactored.py)**

### **Khi nào dùng:**

- Muốn **đánh giá và so sánh nhiều model**
- Cần **báo cáo performance chi tiết**
- **Production evaluation** với workflow hoàn chỉnh
- **Top queries analysis** với full content

### **Workflow hoàn chỉnh:**

```bash
# Bước 1: Chunking dữ liệu luật (tạo chunks.json)
python chunking.py --category BDS  # hoặc --category QDS, v.v.

# Bước 2: Chạy benchmark evaluation hoàn chỉnh
python main_refactored.py

# Output files:
# - results/embedding_evaluation_results.json: Kết quả đánh giá đầy đủ
# - results/top_queries_analysis.json: Top queries analysis sạch
# - Console: Báo cáo ranking + Top 3 queries với full content
```

### **Chi tiết từng bước:**

#### **2.1 Chunking Documents**

#### **Cách 1: Sử dụng chunking.py (dành cho batch processing)**

```bash
# Chunk theo category cụ thể
python chunking.py --category BDS --AI --verbose
python chunking.py --category QDS --AI --verbose

# Chunk một file cụ thể với metadata đầy đủ
python chunking.py --file "path/to/luat.docx" \
  --law-no "52/2014/QH13" \
  --issued-date "2014-06-19" \
  --effective-date "2014-06-19" \
  --signer "Chủ tịch Quốc hội Nguyễn Sinh Hùng"
# Chunk tất cả categories
python chunking.py

# Output: data/BDS_chunk_TIMESTAMP.json, data/QDS_chunk_TIMESTAMP.json, v.v.
```

#### **Cách 2: Sử dụng chunk_and_save.py (KHUYẾN NGHỊ) ⭐**

**Config trong code - Chunk 1 hoặc nhiều files:**

**Bước 1: Mở `chunk_and_save.py` và config:**

```python
# Dòng 38-40: Config files cần chunk
DEFAULT_FILES = [
    "law_content/Bất động sản/Luật 1.docx",
    "law_content/Bất động sản/Luật 2.docx",
    "law_content/Bất động sản/Luật 3.docx",
]

# Dòng 128: Config category
--category = "BDS"

# Dòng 135-143: Config metadata (law_no, law_title, issued_date, ...)
```

**Bước 2: Chạy:**

```bash
python chunk_and_save.py
# Output: data/BDS/Luat_1_chunk_*.json, Luat_2_chunk_*.json, ...
```

**Ưu điểm:**

- ✅ **Config sẵn trong code**: Không cần gõ command dài
- ✅ **Chunk nhiều files**: Chỉ cần 1 lệnh
- ✅ **Auto subfolder**: Tự động tạo `data/{CATEGORY}/`
- ✅ **Metadata đầy đủ**: Config sẵn thông tin luật
- ✅ **AI Review optional**: Thêm `--AI --verbose`

**Tùy chọn AI Review:**

```bash
# AI review cơ bản
python chunk_and_save.py --AI --verbose

# AI review với tùy chỉnh sampling
python chunk_and_save.py --AI --max-files-sample 1 --max-chunks-sample 30 --sample-excerpts 1500 --verbose

# Chỉ lưu nếu AI xác nhận OK
python chunk_and_save.py --AI --strict-ok-only --verbose

# AI review với API key tùy chỉnh
python chunk_and_save.py --AI --api-key "your_gemini_api_key" --verbose
```

#### **Bước 2.3: Merge Chunks (KHUYẾN NGHỊ trước khi upload)**

**Config trong code - Merge tất cả chunks:**

**Mở `merge_chunks.py` và config:**

```python
# Dòng 40-42: Config directory chứa chunks
DEFAULT_DIRECTORY = "data/BDS"
DEFAULT_CATEGORY = "BDS"
DEFAULT_PATTERN = "*_chunk_*.json"
```

**Chạy:**

```bash
python merge_chunks.py
# Output: data/BDS_merged_chunk_HHMMSS_DDMMYY.json
```

**Ưu điểm:**

- ✅ **Dễ quản lý**: 1 file thay vì nhiều files
- ✅ **Upload 1 lần**: Nhanh hơn upload nhiều lần
- ✅ **Remove duplicates**: Tự động loại bỏ chunks trùng
- ✅ **Config sẵn**: Không cần gõ command dài

#### **2.2 Benchmark Evaluation**

```bash
# Chạy evaluation với 4 models + 50 queries random
python main_refactored.py
```

**Script sẽ tự động:**

- ✅ Load chunks từ `data/chunks.json`
- ✅ Load 50 queries benchmark ngẫu nhiên
- ✅ Đánh giá 4 models: minhquan6203, sentence-transformers, namnguyenba2003, truro7
- ✅ Embed documents và upload lên Qdrant
- ✅ Search và tính metrics cho mỗi query
- ✅ Tạo báo cáo ranking theo performance
- ✅ Hiển thị Top 3 queries với full content + thông tin đối chiếu

---

## 📤 **EMBED & UPLOAD (embed_and_upload.py)**

### **Khi nào dùng:**

- Sau khi đã **chunk và merge** chunks
- Muốn **upload lên Qdrant** để sử dụng
- **Test model mới** trước khi đánh giá
- **Thêm data mới** vào collection đã có

### **Workflow đơn giản:**

**Bước 1: Config trong `embed_and_upload.py`:**

```python
# Dòng 74: Config file merged
--chunk-file = "data/BDS_merged_chunk_162134_141025.json"

# Dòng 81: Config model embedding
--model = "minhquan6203/paraphrase-vietnamese-law"

# Dòng 88: Config category
--category = "BDS"
```

**Bước 2: Chạy:**

```bash
python embed_and_upload.py
# Collection name: paraphrase-vietnamese-law-BDS
```

### **Workflow hoàn chỉnh từ đầu đến cuối:**

```bash
# BƯỚC 1: Config và chunk
# Sửa DEFAULT_FILES trong chunk_and_save.py
python chunk_and_save.py

# BƯỚC 2: Config và merge
# Sửa DEFAULT_DIRECTORY trong merge_chunks.py
python merge_chunks.py

# BƯỚC 3: Config và upload
# Sửa --chunk-file, --model, --category trong embed_and_upload.py
python embed_and_upload.py
```

### **Tùy chọn nâng cao:**

#### **Append Mode (Thêm vào collection đã có)**

```bash
# Upload lần đầu (tạo collection mới)
python embed_and_upload.py
# Collection: paraphrase-vietnamese-law-BDS (mới)

# Thêm data mới (tự động APPEND - không xóa data cũ)
python embed_and_upload.py --chunk-file "data/BDS_merged_new.json"
# Collection: paraphrase-vietnamese-law-BDS (append thêm)
```

#### **Force Recreate (XÓA data cũ)**

```bash
# Xóa toàn bộ data cũ và tạo lại
python embed_and_upload.py --force-recreate
```

#### **Dry Run & Options**

```bash
# Test không upload
python embed_and_upload.py --dry-run

# Batch size nhỏ hơn (tiết kiệm memory)
python embed_and_upload.py --batch-size 8

# Dùng CPU thay vì GPU
python embed_and_upload.py --device cpu
```

**📋 Upload Modes:**

| Mode                   | Mô tả                           | Khi nào dùng        |
| ---------------------- | ------------------------------- | ------------------- |
| **Default (Append)**   | Thêm vào collection (không xóa) | Upload thêm data ✅ |
| **`--force-recreate`** | XÓA data cũ và tạo lại ⚠️       | Làm lại từ đầu      |

## 📊 Output Format & Results

### **📄 Format Output Chuẩn (hn2014_chunks.json)**

Chunks được tạo ra có format metadata chuẩn:

```json
{
  "id": "LLHNV-D5-K2-a",
  "content": "Điều 5. Bảo vệ chế độ hôn nhân và gia đình Khoản 2. ...",
  "metadata": {
    "law_no": "52/2014/QH13",
    "law_title": "luat_hon_nhan_va_gia_dinh",
    "law_id": "LLHNV",
    "issued_date": "2014-06-19",
    "effective_date": "2014-06-19",
    "expiry_date": null,
    "signer": "Chủ tịch Quốc hội Nguyễn Sinh Hùng",
    "chapter": "Chương I – NHỮNG QUY ĐỊNH CHUNG",
    "article_no": 5,
    "article_title": "Bảo vệ chế độ hôn nhân và gia đình",
    "clause_no": 2,
    "point_letter": "a",
    "exact_citation": "Điều 5 khoản 2 điểm a.",
    "chapter_number": 1,
    "clause_intro": "Cấm các hành vi sau đây:"
  }
}
```

**✅ Metadata đầy đủ:**

- `law_no`, `law_title`, `law_id`, `issued_date`, `effective_date`, `expiry_date`, `signer`
- `chapter`, `article_no`, `article_title`, `exact_citation`, `chapter_number`
- `clause_no`, `point_letter`, `clause_intro` (tùy loại chunk)

**✅ Không có field thừa:**

- Đã loại bỏ `source_file`, `source_category`, `source_file_name`, `chunk_index`

### **Luồng 1 Output: Benchmark Report**

```bash
# Console output mẫu:
🔝 TOP 3 QUERIES BY TOTAL SCORE (Best Model)

1. Total Score: 12.456
   Query: Dự án đầu tư xây dựng khu đô thị phải có công năng hỗn hợp?
   Category: Bất động sản - Luật Đất Đai
   Expected Answer: Phải có công năng hỗn hợp, đồng bộ hạ tầng kỹ thuật...

   📄 Rank 1: Score 0.823 | Law: LDATDAI
   📝 Citation: Điều 1
   📖 Content: Điều 1. Phạm vi điều chỉnh Luật này quy định...
   ... (500 chars total)
```

### **JSON Output Files**

- **`results/embedding_evaluation_results.json`**: Kết quả đánh giá đầy đủ
- **`results/top_queries_analysis.json`**: Top queries analysis sạch

```json
{
  "model_name": "minhquan6203/paraphrase-vietnamese-law",
  "top_queries_by_total_score": [
    {
      "rank": 1,
      "query": "Câu hỏi benchmark...",
      "total_score": 12.456,
      "question_info": {
        "category": "Bất động sản - Luật Đất Đai",
        "positive_answer": "Câu trả lời tích cực...",
        "source_file": "Luật đất đai 2024.xlsx"
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
```

---

## 🤖 Models & Dataset

### **4 Models được đánh giá:**

1. **🏆 minhquan6203/paraphrase-vietnamese-law** (Best - Fine-tuned cho luật VN)
2. **sentence-transformers/paraphrase-multilingual-mpnet-base-v2** (Multilingual base)
3. **namnguyenba2003/Vietnamese_Law_Embedding_finetuned_v3_256dims** (256-dim custom)
4. **truro7/vn-law-embedding** (SentenceTransformers VN law)

### **Dataset:**

- **Luật Documents**: BDS (Bất động sản), DN (Doanh nghiệp), TM (Thương mại), QDS (Quyền dân sự)
- **Chunking**: Theo cấu trúc Điều-Khoản-Điểm với law_id tự động
- **Benchmark Questions**: 511 câu hỏi từ Excel → Random 50 cho evaluation

### **Law ID Mapping:**

- `LKBDS` → Luật Kinh Doanh Bất Động Sản
- `LNHAO` → Luật Nhà Ở
- `LDATDAI` → Luật Đất Đai
- `LDAUTU` → Luật Đầu Tư
- `LXAYDUNG` → Luật Xây Dựng

---

## 🔧 Cài đặt & Chạy

### **1. Setup Environment**

```bash
# Tạo virtual environment
python -m venv legal_ai_env
source legal_ai_env/bin/activate  # Linux/Mac
# hoặc: legal_ai_env\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Setup Qdrant (nếu chưa có)
# Đảm bảo QDRANT_URL và QDRANT_API_KEY trong .env
```

### **2. Chạy nhanh (Quick Start)**

#### **Luồng Benchmark (Khuyến nghị)**

```bash
# Workflow đánh giá model
python find_law_files.py                    # Tìm files luật
python find_question_files.py              # Trích xuất questions
python chunking.py --category BDS          # Chunk theo batch
python main_refactored.py                  # Benchmark evaluation
```

#### **Luồng Chunking & Upload (Đơn giản) ⭐**

```bash
# Bước 1: Config DEFAULT_FILES trong chunk_and_save.py → Chạy
python chunk_and_save.py

# Bước 2: Config DEFAULT_DIRECTORY trong merge_chunks.py → Chạy
python merge_chunks.py

# Bước 3: Config trong embed_and_upload.py → Chạy
python embed_and_upload.py
```

**Output:**

- `data/{CATEGORY}/*.json` → Chunk files
- `data/{CATEGORY}_merged_*.json` → Merged file
- Qdrant collection: `{model}-{category}`

---

## 📊 Evaluation Metrics

- **Max Score**: Điểm similarity cao nhất trong top-15
- **Average Top-5**: Trung bình điểm của 5 kết quả cao nhất
- **Above 0.7/0.5**: Số kết quả có điểm > 0.7 và > 0.5
- **Cosine Similarity**: Phương pháp đo độ tương đồng
- **Top Queries Analysis**: Phân tích 3 câu hỏi có tổng điểm cao nhất

---

## 🛠️ Utility Scripts

### **Interactive Testing**

```bash
# Test query với model đã upload
python utilities/test_query.py

# Xem top queries analysis
python utilities/view_top_queries.py
```

### **Debug Tools**

```bash
# Debug law_id generation
python test_debug_law_id.py

# Test single file chunking
python test_single_file_chunking.py
```

---

## 📁 Cấu trúc Project

```
vn-law-embedding-benchmark/
├── 📂 data/                          # Chunks đã xử lý
│   ├── BDS/                          # Category folder
│   │   ├── Luat_1_chunk_*.json      # Chunk individual
│   │   └── Luat_2_chunk_*.json
│   └── BDS_merged_chunk_*.json      # Merged file
├── 📂 data_files/                    # Metadata files
│   ├── law_file_paths.json          # Danh sách files luật
│   ├── law_questions.json           # 511 câu hỏi benchmark
│   └── [BDS|DN|TM|QDS]/             # Files theo category
├── 📂 results/                       # Kết quả evaluation
├── 📂 modules/                       # Core modules
│   ├── chunking.py                  # Logic chunking
│   ├── embedding_models.py          # Model management
│   ├── evaluation.py                # Evaluation logic
│   └── qdrant_manager.py            # Qdrant operations
├── 📂 utilities/                     # Helper scripts
├── 🔬 main_refactored.py            # Benchmark evaluation
├── 🧩 chunking.py                   # Batch chunking
├── 📝 chunk_and_save.py             # BƯỚC 1: Chunk files ⭐
├── 🔄 merge_chunks.py               # BƯỚC 2: Merge chunks ⭐
├── 📤 embed_and_upload.py           # BƯỚC 3: Upload to Qdrant ⭐
└── 📋 README.md                     # Documentation
```

---

---

## 🔧 **Chunking với AI Review & Tính năng nâng cao**

### **Tùy chọn AI Review:**

```bash
🤖 Tùy chọn AI Review:
--AI: Bật Gemini AI thẩm định (default: OFF)
--max-files-sample N: Files lấy raw text (default: 2)
--max-chunks-sample N: Chunks gửi AI (default: 50)
--sample-excerpts N: Ký tự excerpts (default: 2000)
--api-key "key": Gemini API key
--strict-ok-only: Chỉ ghi nếu AI "ok"
🔍 Tùy chọn kiểm tra:
--verbose / -v: Chi tiết tiến trình
--validate: Validate chunks quality
--dry-run: Test không ghi file
```

```bash
# Chunk + AI review cơ bản
python chunking.py --category BDS --AI

# Chunk + AI review với tùy chỉnh sampling
python chunking.py --category BDS --AI --max-files-sample 2 --max-chunks-sample 50

# Chunk + AI review với verbose output
python chunking.py --category BDS --AI --verbose

# Chunk + validation (check quality chunks)
python chunking.py --category BDS --validate

# Dry run (test mà không ghi file)
python chunking.py --category BDS --dry-run --validate
```

### **🆔 Tính năng thông minh tạo Law ID:**

Script tự động tạo ID pháp luật thông minh:

- **Luật gốc**: `Luật Sở hữu trí tuệ` → `LSHTT`
- **Luật sửa đổi**: `Luật sửa đổi Luật Sở hữu trí tuệ` → `LSĐBSLSHTT`
- **Luật bổ sung**: `Luật bổ sung Luật Kinh doanh BĐS` → `LSĐBSLKBDS`

Điều này giúp phân biệt rõ ràng luật gốc và luật sửa đổi/bổ sung!

### **📄 Format Output Chuẩn:**

- ✅ **Metadata đầy đủ** với thông tin luật pháp chính xác
- ✅ **Không có field thừa** - clean và consistent
- ✅ **Match format** với `hn2014_chunks.json` chuẩn
- ✅ **Tương thích** với tất cả downstream processing

### **Embedding và Upload lên Qdrant**

**Workflow đơn giản:**

```bash
# 1. Config trong embed_and_upload.py (dòng 74, 81, 88)
--chunk-file = "data/BDS_merged_chunk_162134_141025.json"
--model = "minhquan6203/paraphrase-vietnamese-law"
--category = "BDS"

# 2. Chạy
python embed_and_upload.py
# → Collection: paraphrase-vietnamese-law-BDS
```

**Thêm data mới (Append mode):**

```bash
# Upload thêm data vào collection đã có (không xóa cũ)
python embed_and_upload.py --chunk-file "data/BDS_merged_new.json"
```

**Làm lại từ đầu (Force recreate):**

```bash
# Xóa data cũ và tạo lại
python embed_and_upload.py --force-recreate
```

### **Collection Naming:**

- Format: `{model-name}-{category}`
- Ví dụ: `paraphrase-vietnamese-law-BDS`
- Tự động clean special characters

### **Models hỗ trợ:**

- `minhquan6203/paraphrase-vietnamese-law` ⭐
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- `namnguyenba2003/Vietnamese_Law_Embedding_finetuned_v3_256dims`
- `truro7/vn-law-embedding`

---

## 🏗️ **Modules & Architecture**

### **Luồng thực hiện chi tiết:**

#### **Version Refactored:**

1. **Setup Environment** - Kiểm tra GPU, import libraries
2. **Load Chunks** - Load từ `data/chunks.json` (tạo bởi `chunking.py`)
3. **Load Benchmark Queries** - Load từ `data_files/law_questions.json`
4. **Evaluate Models** - Đánh giá từng model embedding
5. **Generate Report** - Tạo báo cáo chi tiết

#### **Luồng chuẩn bị data:**

1. **`find_law_files.py`** → Tạo `data_files/law_file_paths.json`
2. **`find_question_files.py`** → Tạo `data_files/law_questions.json`
3. **`chunking.py`** → Load từ `law_file_paths.json`, chunk và tạo `data/chunks.json`
4. **`main_refactored.py`** → Load `chunks.json` và chạy evaluation

### **Chi tiết Modules:**

#### **`chunking.py`**

- `chunk_law_document()`: Chia văn bản luật thành chunks
- `read_docx()`: Đọc file .docx/.doc
- `normalize_lines()`: Chuẩn hóa text
- `generate_law_id()`: Tạo ID cho luật
- `build_review_payload()`: **MỚI** - Sampling thông minh để AI review
- `call_gemini_review()`: Gọi Gemini để thẩm định chunks
- **✅ Command line arguments** cho `--law-no`, `--issued-date`, `--effective-date`, `--signer`

**Chiến lược Sampling cho AI Review:**

1. **Raw Text Sampling**: Chỉ lấy 2 files đầu, mỗi file 5k chars excerpts
2. **Chunks Sampling**: Sample 50 chunks (17 mỗi loại: article/clause/point)
3. **Smart Excerpts**: Chia đều chars cho từng file, sample 3 phần (đầu/giừa/cuối)

Điều này giảm payload từ ~100KB xuống ~10-15KB, tránh vượt token limit!

#### **`data_loader.py`**

- `load_all_law_documents()`: Load tất cả documents
- `load_question_benchmark()`: Load benchmark queries
- `save_chunks_to_json()`: Lưu chunks ra JSON

#### **`qdrant_manager.py`**

- `get_qdrant_client()`: Kết nối Qdrant
- `ensure_collection()`: Tạo collection
- `upsert_embeddings_to_qdrant()`: Upload embeddings
- `search_qdrant()`: Tìm kiếm

#### **`embedding_models.py`**

- `encode_with_transformers()`: Encode với Transformers
- `encode_with_sentence_transformers()`: Encode với Sentence Transformers
- `encode_texts()`: Factory function

#### **`evaluation.py`**

- `evaluate_single_model()`: Đánh giá 1 model
- `run_evaluation_all_models()`: Đánh giá tất cả models
- `calculate_metrics()`: Tính toán metrics

---

---

## 🔄 **Migration Guide**

Để migrate từ version gốc sang refactor:

```python
# Thay vì import tất cả từ embedding_evaluation.py
from modules.chunking import chunk_law_document
from modules.data_loader import load_all_law_documents
from modules.evaluation import evaluate_single_model

# Sử dụng các functions độc lập
```

---

## 🤝 **Contributing**

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request
