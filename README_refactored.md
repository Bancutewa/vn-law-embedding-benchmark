# Vietnamese Law Embedding Benchmark - Refactored Version

Phiên bản tái cấu trúc của dự án đánh giá embedding cho luật tiếng Việt với code được tổ chức thành modules.

## 🚀 Cập nhật luồng chạy mới (24/09/2025)

### Luồng mới với tổ chức theo bộ luật

#### Bước 1: Tạo danh sách file luật theo category

```bash
python find_law_files.py
```

- Tạo file tổng hợp: `data_files/law_file_paths.json`
- Tạo file riêng biệt theo category:
  - `data_files/BDS/bds_file_paths.json` (Bất động sản - 7 files)
  - `data_files/DN/dn_file_paths.json` (Doanh nghiệp - 1 file)
  - `data_files/TM/tm_file_paths.json` (Thương mại - 1 file)
  - `data_files/QDS/qds_file_paths.json` (Quyền dân sự - 4 files)

#### Bước 2: Chunk theo category cụ thể hoặc file đơn lẻ

```bash
# Chunk bộ luật Bất động sản
python chunking.py --category BDS

# Chunk bộ luật Quyền dân sự
python chunking.py --category QDS

# Chunk một file cụ thể
python chunking.py --file "law_content/Quyền dân sự_/Luật nuôi con nuôi/Luật_/Luật nuôi con nuôi.docx"

# Chunk tất cả (tương thích ngược)
python chunking.py
```

#### Bước 3: Chunk với AI review và tùy chọn nâng cao

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

### 📋 **Giải thích các tùy chọn:**

#### **Tùy chọn cơ bản:**

- `--category BDS/DN/TM/QDS`: Chunk theo bộ luật cụ thể
- `--file "path/to/file.docx"`: Chunk một file duy nhất
- Không chỉ định gì: Chunk tất cả categories

#### **🆔 Tính năng thông minh tạo Law ID:**

Script tự động tạo ID pháp luật thông minh:

- **Luật gốc**: `Luật Sở hữu trí tuệ` → `LSHTT`
- **Luật sửa đổi**: `Luật sửa đổi Luật Sở hữu trí tuệ` → `LSĐBSLSHTT`
- **Luật bổ sung**: `Luật bổ sung Luật Kinh doanh BĐS` → `LSĐBSLKBDS`

Điều này giúp phân biệt rõ ràng luật gốc và luật sửa đổi/bổ sung!

#### **🤖 Tùy chọn AI Review:**

- `--AI`: Bật Gemini AI để thẩm định chất lượng chunks
- `--max-files-sample N`: Số files tối đa lấy raw text (default: 2)
- `--max-chunks-sample N`: Số chunks tối đa gửi AI review (default: 50)
- `--sample-excerpts N`: Tổng ký tự excerpts từ files (default: 2000)
- `--api-key "your_key"`: Gemini API key (hoặc set env var)
- `--strict-ok-only`: Chỉ ghi chunks nếu AI xác nhận "ok"

#### **Tùy chọn kiểm tra và debug:**

- `--verbose` / `-v`: In chi tiết tiến trình chunking và AI review
- `--validate`: Validate chunks sau khi tạo (check format, metadata)
- `--dry-run`: Test mode - chạy nhưng không ghi file output

#### **Ví dụ workflow với kiểm tra:**

```bash
# 1. Test validation trước
python chunking.py --category BDS --validate --dry-run --verbose

# 2. Chunk với AI review chi tiết
python chunking.py --category BDS --AI --verbose --max-files-sample 3

# 3. Chunk và validate quality
python chunking.py --category BDS --AI --validate --strict-ok-only
```

#### Bước 4: Embedding và upload lên Qdrant

Sau khi có file chunks, embed và upload lên Qdrant:

```bash
# Embed BDS chunks với model minhquan6203
python embed_and_upload.py \
  --chunk-file "data/BDS_chunk_155638_250925.json" \
  --model "minhquan6203/paraphrase-vietnamese-law" \
  --category "BDS"

# Embed QDS chunks với model khác
python embed_and_upload.py \
  --chunk-file "data/QDS_chunk_151649_250925.json" \
  --model "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" \
  --category "QDS"

# Embed TM chunks (dry run để test)
python embed_and_upload.py \
  --chunk-file "data/TM_chunk_154949_250925.json" \
  --model "namnguyenba2003/Vietnamese_Law_Embedding_finetuned_v3_256dims" \
  --category "TM" \
  --dry-run
```

### 🔧 **Cấu hình script embed_and_upload.py:**

#### **Tham số chính (sửa trong code):**

```python
# Trong embed_and_upload.py - dòng 73-88
parser.add_argument("--chunk-file", default="data/BDS_chunk_155638_250925.json")  # ←←← SỬA PATH
parser.add_argument("--model", default="minhquan6203/paraphrase-vietnamese-law") # ←←← SỬA MODEL
parser.add_argument("--category", default="BDS")                                # ←←← SỬA CATEGORY
```

#### **Collection naming:**

- Format: `model-name-category`
- Ví dụ: `minhquan6203-paraphrase-vietnamese-law-BDS`
- Tự động clean special characters

#### **Models hỗ trợ:**

- **Transformers**: `minhquan6203/paraphrase-vietnamese-law`, `namnguyenba2003/...`
- **Sentence Transformers**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

#### **Ví dụ tái sử dụng:**

```bash
# Cho QDS
python embed_and_upload.py --chunk-file "data/QDS_chunk_xxx.json" --category "QDS"

# Cho TM
python embed_and_upload.py --chunk-file "data/TM_chunk_xxx.json" --category "TM"

# Model khác
python embed_and_upload.py --model "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
```

### File output

- Chunk theo category: `data/[CATEGORY]_chunk_[timestamp].json`
- Ví dụ: `data/BDS_chunk_174902_240925.json`, `data/QDS_chunk_174907_240925.json`

### Ưu điểm

- ✅ Tổ chức file theo bộ luật rõ ràng
- ✅ Dễ quản lý và import lên Qdrant theo category
- ✅ Tương thích ngược với luồng cũ
- ✅ Chunk từng category một để kiểm soát tốt hơn
- ✅ Chunk file đơn lẻ để test và debug nhanh
- ✅ File paths được tách riêng theo từng bộ luật

## 🏗️ Cấu trúc dự án (Refactored)

```
vn-law-embedding-benchmark/
├── modules/                    # Các modules đã tách ra
│   ├── __init__.py
│   ├── chunking.py            # Logic chunking luật VN
│   ├── data_loader.py         # Load documents và queries
│   ├── qdrant_manager.py      # Quản lý Qdrant operations
│   ├── embedding_models.py    # Các model embedding
│   └── evaluation.py          # Logic đánh giá
├── data/                      # JSON outputs từ chunking
├── main_refactored.py         # Main script refactor
├── embedding_evaluation.py    # Script gốc (1900+ lines)
├── test_query.py              # Script test query
├── requirements.txt
└── README_refactored.md       # File này
```

## 🚀 Cách sử dụng

### 1. Chuẩn bị môi trường

```bash
# Clone repository
git clone <repository-url>
cd vn-law-embedding-benchmark

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Thiết lập biến môi trường
cp .env.example .env
# Edit .env với QDRANT_URL và QDRANT_API_KEY
```

### 2. Chuẩn bị dữ liệu

```bash
# Bước 1: Tạo danh sách file luật
python find_law_files.py

# Bước 2: Tạo danh sách câu hỏi
python find_question_files.py

# Bước 3: Chunk tất cả documents (tạo data/chunks.json)
python chunking.py

# Chunk với AI review (tùy chọn):
python chunking.py --AI --max-files-sample 2 --max-chunks-sample 50
```

### 3. Chạy đánh giá (Refactored Version)

```bash
# Chạy version refactor (sẽ load chunks.json và evaluate)
python main_refactored.py
```

### 4. Chạy đánh giá (Original Version)

```bash
# Chạy version gốc
python embedding_evaluation.py
```

## 📊 Luồng thực hiện

### Version Refactored:

1. **Setup Environment** - Kiểm tra GPU, import libraries
2. **Load Chunks** - Load từ `data/chunks.json` (tạo bởi `chunking.py`)
3. **Load Benchmark Queries** - Load từ `data_files/law_questions.json`
4. **Evaluate Models** - Đánh giá từng model embedding
5. **Generate Report** - Tạo báo cáo chi tiết

### Luồng chuẩn bị data:

1. **`find_law_files.py`** → Tạo `data_files/law_file_paths.json`
2. **`find_question_files.py`** → Tạo `data_files/law_questions.json`
3. **`chunking.py`** → Load từ `law_file_paths.json`, chunk và tạo `data/chunks.json`
4. **`main_refactored.py`** → Load `chunks.json` và chạy evaluation

### Version Original:

- Tất cả logic trong 1 file `embedding_evaluation.py` (1906 lines)

## 🤖 Models được đánh giá

| Model                                                           | Type                  | Description                       |
| --------------------------------------------------------------- | --------------------- | --------------------------------- |
| `minhquan6203/paraphrase-vietnamese-law`                        | transformers          | Fine-tuned cho luật VN            |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`   | sentence-transformers | Base model đa ngôn ngữ            |
| `namnguyenba2003/Vietnamese_Law_Embedding_finetuned_v3_256dims` | transformers          | 256 dimensions                    |
| `truro7/vn-law-embedding`                                       | sentence_transformers | Sentence Transformers cho luật VN |

## 📈 Metrics đánh giá

- **Max Score**: Điểm cao nhất cho mỗi query
- **Top-5 Average**: Trung bình 5 kết quả cao nhất
- **Scores ≥ 0.7**: Số kết quả có độ tương đồng > 0.7
- **Scores ≥ 0.5**: Số kết quả có độ tương đồng > 0.5

## 🛠️ Modules

### `chunking.py`

- `chunk_law_document()`: Chia văn bản luật thành chunks
- `read_docx()`: Đọc file .docx/.doc
- `normalize_lines()`: Chuẩn hóa text
- `generate_law_id()`: Tạo ID cho luật
- `build_review_payload()`: **MỚI** - Sampling thông minh để AI review
- `call_gemini_review()`: Gọi Gemini để thẩm định chunks

#### Chiến lược Sampling cho AI Review:

1. **Raw Text Sampling**: Chỉ lấy 2 files đầu, mỗi file 5k chars excerpts
2. **Chunks Sampling**: Sample 50 chunks (17 mỗi loại: article/clause/point)
3. **Smart Excerpts**: Chia đều chars cho từng file, sample 3 phần (đầu/giừa/cuối)

Điều này giảm payload từ ~100KB xuống ~10-15KB, tránh vượt token limit!

### `data_loader.py`

- `load_all_law_documents()`: Load tất cả documents
- `load_question_benchmark()`: Load benchmark queries
- `save_chunks_to_json()`: Lưu chunks ra JSON

### `qdrant_manager.py`

- `get_qdrant_client()`: Kết nối Qdrant
- `ensure_collection()`: Tạo collection
- `upsert_embeddings_to_qdrant()`: Upload embeddings
- `search_qdrant()`: Tìm kiếm

### `embedding_models.py`

- `encode_with_transformers()`: Encode với Transformers
- `encode_with_sentence_transformers()`: Encode với Sentence Transformers
- `encode_texts()`: Factory function

### `evaluation.py`

- `evaluate_single_model()`: Đánh giá 1 model
- `run_evaluation_all_models()`: Đánh giá tất cả models
- `calculate_metrics()`: Tính toán metrics

## 🔍 Test Query

```bash
# Chạy interactive query tool
python test_query.py
```

## 📁 Outputs

- `data_files/law_file_paths.json`: Danh sách file luật (từ `find_law_files.py`)
- `data_files/law_questions.json`: Benchmark questions (từ `find_question_files.py`)
- `data/chunks.json`: Chunks đã được xử lý (từ `chunking.py`)
- `results/embedding_evaluation_results.json`: Kết quả đánh giá chi tiết
- `results/top_queries_analysis.json`: Phân tích queries top

## ⚡ Lợi ích của Version Refactored

1. **Modularity**: Code được chia thành modules nhỏ, dễ maintain
2. **Reusability**: Có thể import và sử dụng từng module độc lập
3. **Testability**: Dễ viết unit tests cho từng module
4. **Readability**: Code dễ đọc, hiểu và debug
5. **Extensibility**: Dễ thêm features mới

## 🔄 Migration Guide

Để migrate từ version gốc sang refactor:

```python
# Thay vì import tất cả từ embedding_evaluation.py
from modules.chunking import chunk_law_document
from modules.data_loader import load_all_law_documents
from modules.evaluation import evaluate_single_model

# Sử dụng các functions độc lập
```

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## 📄 License

MIT License
