# 🔍 Vietnamese Legal Embedding Model Evaluation & Testing

Dự án này đánh giá và so sánh các mô hình embedding cho văn bản luật tiếng Việt, với giao diện Gradio để test interactive.

## 📋 Tổng quan

- **Dataset:** Luật Hôn nhân và Gia đình Việt Nam 2014 (396 chunks)
- **Models:** 6 mô hình embedding Vietnamese và multilingual
- **Chunking:** 2-pass system với clause intro injection
- **Storage:** RAM-based caching (không dùng vector database)
- **Interface:** Gradio web interface cho testing real-time

## 🚀 Cách sử dụng

### 1. Cài đặt môi trường

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

- ✅ Load và chunk dữ liệu luật từ DOCX
- ✅ Đánh giá 6 mô hình embedding
- ✅ Tạo báo cáo chi tiết với ranking

## 🤖 Models được đánh giá

1. **🏆 minhquan6203/paraphrase-vietnamese-law** (Best)
2. **huyhuy123/paraphrase-vietnamese-law-ALQAC**
3. **namnguyenba2003/Vietnamese_Law_Embedding_finetuned_v3_256dims**
4. **sentence-transformers/paraphrase-multilingual-mpnet-base-v2**
5. **BAAI/bge-m3**
6. **maiduchuy321/vietnamese-bi-encoder-fine-tuning-for-law-chatbot**

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
