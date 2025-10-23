# 📦 Chunking Data - Vietnamese Legal Document Processing

> Hệ thống chunk, embed và upload văn bản luật Việt Nam lên Qdrant vector database

---

## 🎯 Tóm Tắt

1. **Tìm files** luật trong thư mục → JSON catalog
2. **Chunk** văn bản luật theo cấu trúc (Chương > Điều > Khoản > Điểm) → JSON chunks
3. **Merge** nhiều chunk files → 1 file lớn
4. **Embed** chunks thành vectors → Upload lên Qdrant

---

## 📁 Cấu Trúc Thư Mục

```
Chunking_Data/
│
├── 📦 core/                          # Thuật toán cốt lõi (pure functions)
│   ├── docx_reader.py                # Đọc .doc/.docx (5 fallback methods)
│   ├── law_chunker.py                # Chunking logic chính (2-pass parsing)
│   ├── law_id_generator.py           # Tự động tạo law_id từ tên file
│   └── __init__.py
│
├── 🔄 pipeline/                      # Workflows (high-level)
│   ├── file_discovery.py             # Tìm và catalog files luật
│   ├── chunking_pipeline.py          # Batch chunking workflow
│   ├── embedding_pipeline.py         # Embedding & upload workflow
│   └── __init__.py
│
├── 💾 storage/                       # Data persistence
│   ├── json_handler.py               # Save/load/merge JSON
│   ├── qdrant_client.py              # Qdrant operations (connect, upload, embed)
│   └── __init__.py
│
├── 🔧 scripts/                       # CLI tools (user-facing)
│   ├── find_files.py                 # Tìm tất cả files luật
│   ├── chunk_documents.py            # Chunk văn bản → JSON
│   ├── merge_chunks.py               # Merge nhiều JSON → 1 file
│   ├── upload_qdrant.py              # Upload chunks lên Qdrant
│   └── __init__.py
│
├── ⚙️ config.py                      # Cấu hình chung (models, paths, categories)
├── 📖 README.md                      # File này
├── 🏗️ ARCHITECTURE.md                # Chi tiết kiến trúc & design decisions
├── 📊 COMPARISON_REPORT.md           # So sánh code cũ vs mới
├── 🌍 env.example                    # Mẫu file .env
├── __init__.py                       # Package initialization
└── __main__.py                       # Entry point (python -m Chunking_Data)
```

---

## Workflows

```bash
# Bước 1: Tìm tất cả files luật
python -m Chunking_Data.scripts.find_files
# Output: data_files/BDS/bds_file_paths.json, DN/dn_file_paths.json, ...

# Bước 2: Chunk theo category
python -m Chunking_Data.scripts.chunk_documents --category BDS
# Output: data/BDS/file1_chunk_*.json, file2_chunk_*.json, ...

# Chunk 1 file luật thành chunks
python -m Chunking_Data.scripts.chunk_documents --file "law_content/Bất động sản/Luật Nhà ở/Luật Nhà ở 2023.docx"

# Bước 3 (Optional): Merge chunks
# Merge trong 1 folder
python -m Chunking_Data.scripts.merge_chunks --directory data/BDS --category BDS
# Merge nhiều files
python -m Chunking_Data.scripts.merge_chunks --directory data/BDS --files file1.json file2.json  # Merge files cụ thể
# Output: data/BDS_merged_chunk_HHMMSS_DDMMYY.json

# Bước 4: Upload lên Qdrant
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file "data/BDS_merged_chunk_*.json" \
    --category BDS
# Output: Qdrant collection: paraphrase-vietnamese-law-BDS
```

**Giải thích**:

- **Bước 1**: Scan toàn bộ `law_content/`, phân loại theo category (BDS, DN, TM, QDS)
- **Bước 2**: Chunk tất cả files trong category → Nhiều JSON files
- **Bước 3**: Gộp thành 1 file lớn (dễ quản lý, nhanh hơn khi upload)
- **Bước 4**: Embed + Upload lên Qdrant

**Phù hợp**: Production, chunk nhiều files cùng lúc

---

## 📚 Giải Thích Chi Tiết Từng File

### 📦 Core Layer (Thuật toán thuần túy)

#### `docx_reader.py`

**Chức năng**: Đọc file .doc/.docx
**Methods**:

- `read_docx(file_path)` → raw text
- Fallback chain: python-docx → docx2txt → textract → pypandoc → antiword

**Khi nào dùng**: Import trực tiếp khi cần đọc docx

```python
from Chunking_Data.core.docx_reader import read_docx
text = read_docx("my_law.docx")
```

---

#### `law_chunker.py`

**Chức năng**: Chunking logic chính
**Algorithm**:

1. **Pass 1 (Pre-scan)**: Scan để biết có bao nhiêu Chương/Điều
2. **Pass 2 (Strict parsing)**: Parse từng dòng theo state machine
3. **Flush**: Tạo chunks khi đóng Khoản/Điểm/Điều

**Main function**:

```python
chunk_law_document(text, law_id, law_no, law_title, ...)
→ (chunks, stats)
```

**Khi nào dùng**: Core function, được gọi bởi ChunkingPipeline

---

#### `law_id_generator.py`

**Chức năng**: Tự động tạo law_id từ tên file
**Examples**:

- `"Luật Hôn nhân và Gia đình 2014.docx"` → `"LHNVDG"`
- `"Luật Xây dựng 2020.docx"` → `"LXAYDUNG"`
- `"Luật sửa đổi Luật Đất đai.docx"` → `"LSĐBSLDATDAI"`

**Khi nào dùng**: Tự động được gọi trong ChunkingPipeline

---

### 🔄 Pipeline Layer (Workflows)

#### `file_discovery.py`

**Chức năng**: Tìm và catalog files luật
**Functions**:

- `find_law_files(law_content_dir)` → list file paths
- `create_file_paths_list()` → JSON catalog theo category

**Output**: `data_files/BDS/bds_file_paths.json`, ...

---

#### `chunking_pipeline.py`

**Chức năng**: Batch chunking workflow
**Class**: `ChunkingPipeline`
**Methods**:

- `process_single_file(file_path, ...)` → (chunks, stats)
- `process_files(file_paths, ...)` → (all_chunks, summary)

**Khi nào dùng**: Chunk nhiều files

```python
from Chunking_Data.pipeline.chunking_pipeline import ChunkingPipeline

pipeline = ChunkingPipeline(verbose=True)
chunks, stats = pipeline.process_single_file("law.docx")
```

---

#### `embedding_pipeline.py`

**Chức năng**: Embedding & upload workflow
**Class**: `EmbeddingPipeline`
**Methods**:

- `extract_texts(chunks)` → texts
- `encode_texts(texts)` → embeddings
- `upload_to_qdrant(embeddings, chunks, collection_name)`
- `process_and_upload(chunks, category)` → results

**Khi nào dùng**: Upload chunks lên Qdrant

---

### 💾 Storage Layer (Data Persistence)

#### `json_handler.py`

**Functions**:

- `save_chunks_to_json(chunks, file_path)` - Lưu chunks
- `load_chunks_from_json(file_path)` - Đọc chunks
- `merge_chunk_files(file_paths)` - Merge nhiều files
- `get_file_stats(file_path)` - Lấy statistics

---

#### `qdrant_client.py`

**Functions**:

- `get_qdrant_client()` - Kết nối Qdrant (từ .env)
- `ensure_collection(client, name, vector_size)` - Tạo collection
- `encode_texts(texts, model_info, device)` - Embed texts
- `upsert_embeddings_to_qdrant(client, collection, embeddings, chunks)` - Upload

**Khi nào dùng**: Được gọi bởi EmbeddingPipeline

---

### 🔧 Scripts Layer (CLI)

#### `find_files.py`

**Chức năng**: Tìm tất cả files luật trong `law_content/`

```bash
python -m Chunking_Data.scripts.find_files
```

**Output**:

- `data_files/law_file_paths.json` (tất cả)
- `data_files/BDS/bds_file_paths.json` (Bất động sản)
- `data_files/DN/dn_file_paths.json` (Doanh nghiệp)
- `data_files/TM/tm_file_paths.json` (Thương mại)
- `data_files/QDS/qds_file_paths.json` (Quyền dân sự)

---

#### `chunk_documents.py`

**Chức năng**: Chunk văn bản → JSON

**Options**:

```bash
--category BDS              # Chunk category
--file path/to/law.docx     # Chunk 1 file
--all                       # Chunk tất cả
--validate                  # Validate chunks sau khi tạo
--dry-run                   # Test không ghi file
--verbose                   # Log chi tiết
```

**Examples**:

```bash
# Chunk category
python -m Chunking_Data.scripts.chunk_documents --category BDS --validate

# Chunk 1 file với metadata
python -m Chunking_Data.scripts.chunk_documents \
    --file "law.docx" \
    --law-no "52/2014/QH13" \
    --issued-date "2014-06-19" \
    --effective-date "2015-01-01"
```

---

#### `merge_chunks.py`

**Chức năng**: Merge nhiều JSON files → 1 file

**Options**:

```bash
--directory data/BDS        # Merge tất cả trong folder
--files file1.json file2.json  # Merge files cụ thể
--pattern "*_chunk_*.json"  # Pattern tìm files
--category BDS              # Category cho output filename
```

**Examples**:

```bash
# Merge tất cả chunks trong BDS
python -m Chunking_Data.scripts.merge_chunks \
    --directory data/BDS \
    --category BDS
# Output: data/BDS_merged_chunk_HHMMSS_DDMMYY.json
```

---

#### `upload_qdrant.py`

**Chức năng**: Embed + Upload lên Qdrant

**Options**:

```bash
--chunk-file data/chunks.json   # File chunks
--category BDS                  # Category cho collection name
--model MODEL_NAME              # Embedding model (default: paraphrase-vietnamese-law)
--device cuda|cpu               # Device
--batch-size 16                 # Batch size
--force-recreate                # XÓA data cũ và tạo lại
--append                        # Thêm vào collection (default)
--dry-run                       # Test không upload
```

**Examples**:

```bash
# Upload với default model
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file "data/BDS_merged.json" \
    --category BDS

# Upload với model khác
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file "data/BDS.json" \
    --category BDS \
    --model "BAAI/bge-m3" \
    --device cpu
```

**Collection naming**: `{model-name}-{category}`

- Example: `paraphrase-vietnamese-law-BDS`

---

### ⚙️ `config.py`

**Chức năng**: Cấu hình chung

**Constants**:

- `DEFAULT_LAW_CONTENT_DIR = "law_content"`
- `DEFAULT_EMBEDDING_MODEL = "minhquan6203/paraphrase-vietnamese-law"`
- `SUPPORTED_MODELS` - Dictionary các models support
- `CATEGORY_MAPPINGS` - Mapping BDS, DN, TM, QDS

**Functions**:

- `get_qdrant_config()` - Lấy config từ .env
- `get_category_info(code)` - Lấy info category
- `validate_config()` - Validate .env

**Check config**:

```bash
python -m Chunking_Data --config
```

---

## 🌍 Configuration (.env)

Tạo file `.env` trong root:

```bash
# REQUIRED
QDRANT_URL=https://your-qdrant-url.com
QDRANT_API_KEY=your-api-key

# OPTIONAL
EMBEDDING_MODEL=minhquan6203/paraphrase-vietnamese-law
DEVICE=cuda
BATCH_SIZE=16
```

**Validate config**:

```bash
python -m Chunking_Data --validate-config
```

---

## 📂 Categories

| Code  | Tên Đầy Đủ   | Thư Mục                      |
| ----- | ------------ | ---------------------------- |
| `BDS` | Bất động sản | law_content/Bất động sản/    |
| `DN`  | Doanh nghiệp | law*content/Doanh nghiệp*/   |
| `TM`  | Thương mại   | law_content/Luật Thương Mại/ |
| `QDS` | Quyền dân sự | law*content/Quyền dân sự*/   |

---

## 🤖 Supported Models

| Model                                            | Type                  | Dimension | Ghi Chú                |
| ------------------------------------------------ | --------------------- | --------- | ---------------------- |
| `minhquan6203/paraphrase-vietnamese-law`         | transformers          | 768       | **Recommended** cho VN |
| `BAAI/bge-m3`                                    | transformers          | 1024      | Multilingual           |
| `sentence-transformers/paraphrase-multilingual-` | sentence_transformers | 768       | SentenceTransformer    |
| `namnguyenba2003/Vietnamese_Law_Embedding_`      | transformers          | 256       | Vietnamese Law 256-dim |

---

## 🐛 Troubleshooting

### Lỗi: "QDRANT_URL not set"

**Giải pháp**: Tạo file `.env`

```bash
echo "QDRANT_URL=https://your-url.com" > .env
echo "QDRANT_API_KEY=your-key" >> .env
```

### Lỗi: "Vector size mismatch"

**Nguyên nhân**: Collection cũ có dimension khác
**Giải pháp**: Dùng `--force-recreate`

```bash
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file data/BDS.json \
    --category BDS \
    --force-recreate
```

### Lỗi: "CUDA out of memory"

**Giải pháp 1**: Giảm batch size

```bash
--batch-size 8
```

**Giải pháp 2**: Dùng CPU

```bash
--device cpu
```

### Không đọc được file .doc

**Nguyên nhân**: Thiếu dependencies
**Giải pháp**:

```bash
pip install python-docx docx2txt textract pypandoc
# Windows: Cài antiword (cho .doc cũ)
```

---

## 💡 Tips & Best Practices

### 1. Workflow Khuyến Nghị

```bash
# 1. Find files trước (1 lần)
python -m Chunking_Data.scripts.find_files

# 2. Chunk theo category (dễ quản lý)
python -m Chunking_Data.scripts.chunk_documents --category BDS --validate

# 3. Merge trước khi upload (nhanh hơn)
python -m Chunking_Data.scripts.merge_chunks --directory data/BDS --category BDS

# 4. Upload với recommended model
python -m Chunking_Data.scripts.upload_qdrant \
    --chunk-file data/BDS_merged.json \
    --category BDS
```

### 2. Organization

- ✅ Chunk theo category riêng biệt
- ✅ Merge chunks trước khi upload
- ✅ Dùng `--validate` để check quality
- ✅ Backup JSON files

### 3. Performance

- ✅ GPU nếu có (`--device cuda`)
- ✅ Batch size phù hợp (16 cho 8GB VRAM)
- ✅ Merge trước upload (giảm overhead)

---

## 🔗 Xem Thêm

- **ARCHITECTURE.md** - Chi tiết kiến trúc, design decisions
- **COMPARISON_REPORT.md** - So sánh với code cũ
- **env.example** - Mẫu file .env

---

## 📧 Support

Nếu gặp vấn đề:

1. Check **Troubleshooting** section
2. Validate config: `python -m Chunking_Data --validate-config`
3. Đọc error messages (thường có hint)

---

**Made with ❤️ for Vietnamese Legal AI**
