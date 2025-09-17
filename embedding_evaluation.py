#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Đánh Giá Mô Hình Embedding Cho Luật Tiếng Việt
Chuyển đổi từ notebook thành script để chạy một lần
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd
import os
import sys
import re
import json
import time
import gc
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def check_version(package_name, expected_version=None):
    """Kiểm tra phiên bản package"""
    try:
        import importlib.metadata
        version = importlib.metadata.version(package_name)
        if expected_version and version != expected_version:
            print(f"⚠️  {package_name}: expected {expected_version}, got {version}")
        else:
            print(f"✅ {package_name}: {version}")
        return True, version
    except Exception as e:
        print(f"❌ {package_name}: not found or error ({e})")
        return False, None

def setup_environment():
    """Thiết lập môi trường và import các thư viện cần thiết"""
    print("🔄 Importing core libraries...")
    
    # Check critical versions
    check_version("huggingface_hub", "0.16.4")
    check_version("tokenizers", "0.13.3")
    check_version("transformers", "4.32.1")
    check_version("sentence-transformers", "2.2.2")
    
    # Import transformers
    print("\n🔄 Importing transformers...")
    try:
        from transformers import AutoTokenizer, AutoModel
        print("✅ transformers imported successfully")
        
        # Quick functionality test
        test_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        test_tokens = test_tokenizer("test", return_tensors="pt")
        print("✅ transformers functionality test passed")
        del test_tokenizer, test_tokens
        
    except Exception as e:
        print(f"❌ transformers failed: {e}")
        raise
    
    # Import sentence-transformers
    print("\n🔄 Importing sentence-transformers...")
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers imported successfully")
        
        # Quick functionality test
        test_model = SentenceTransformer("all-MiniLM-L6-v2")
        test_embedding = test_model.encode(["test sentence"])
        print(f"✅ sentence-transformers functionality test passed (embedding shape: {test_embedding.shape})")
        del test_model, test_embedding
        
    except Exception as e:
        print(f"❌ sentence-transformers failed: {e}")
        raise
    
    # Import document processing
    print("\n🔄 Importing document processing...")
    try:
        from docx import Document
        print("✅ python-docx imported successfully")
    except ImportError:
        print("📦 Installing python-docx...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-docx==0.8.11"])
        from docx import Document
        print("✅ python-docx installed and imported")
    
    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Using device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("\n🎉 All libraries loaded successfully! Ready for evaluation.")
    return device

# Roman numeral converter
ROMAN_MAP = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}

def roman_to_int(s: str):
    """Chuyển đổi số La Mã sang số nguyên"""
    s = s.upper().strip()
    if not s or any(ch not in ROMAN_MAP for ch in s):
        return None
    total = 0
    prev = 0
    for ch in reversed(s):
        val = ROMAN_MAP[ch]
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return total

def mean_pooling(model_output, attention_mask):
    """Mean pooling để tạo sentence embeddings từ token embeddings"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_with_transformers(texts, model_name, max_length=512, batch_size=32, device="cuda"):
    """Encode texts using transformers library với mean pooling"""
    from transformers import AutoTokenizer, AutoModel
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=max_length,
                return_tensors='pt'
            ).to(device)
            
            # Get model output
            model_output = model(**encoded)
            
            # Mean pooling
            sentence_embeddings = mean_pooling(model_output, encoded['attention_mask'])
            
            # Normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            
            # Move to CPU and convert to numpy
            embeddings = sentence_embeddings.cpu().numpy()
            all_embeddings.append(embeddings)
            
            # Clear GPU memory
            del encoded, model_output, sentence_embeddings, embeddings
            torch.cuda.empty_cache() if device == "cuda" else None
    
    # Delete model to free memory
    del model, tokenizer
    torch.cuda.empty_cache() if device == "cuda" else None
    
    # Combine all embeddings
    final_embeddings = np.vstack(all_embeddings)
    print(f"   ✅ Generated {final_embeddings.shape[0]} embeddings, stored in RAM")
    
    return final_embeddings

def encode_with_sentence_transformers(texts, model_name, batch_size=32, device="cuda"):
    """Encode texts using sentence-transformers library"""
    from sentence_transformers import SentenceTransformer
    
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    model.to(device)
    
    # Encode all texts
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Delete model to free memory
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    print(f"   ✅ Generated {embeddings.shape[0]} embeddings, stored in RAM")
    
    return embeddings

def read_docx(file_path):
    """Đọc file docx và trả về text"""
    from docx import Document
    
    print(f"   Reading file: {file_path}")
    doc = Document(file_path)
    text = "\n".join((p.text or "").strip() for p in doc.paragraphs)
    print(f"   ✅ Successfully read {len(text):,} characters")
    return text

def normalize_lines(text: str):
    """Chuẩn hóa các dòng text - loại bỏ whitespace thừa"""
    lines = [re.sub(r'\s+$', '', ln) for ln in text.splitlines()]
    print(f"   ✅ Normalized {len(lines):,} lines")
    return lines

# ===================== GEMINI EVALUATION AGENT (optional) =====================
# Env controls:
#   CHUNK_AI_REVIEW=true|1        → enable Gemini review for chunks
#   CHUNK_STRICT_OK_ONLY=true|1   → only proceed when Gemini returns status=ok
#   GEMINI_API_KEY                → required when review is enabled

GEMINI_MODEL_NAME = "gemini-2.5-flash"

GEMINI_PROMPT = """Bạn là chuyên gia pháp điển hoá & kiểm thử dữ liệu luật.
Tôi gửi cho bạn:
1) Summary thống kê & cảnh báo từ bộ chunking.
2) Một số "excerpts" (trích đoạn nguyên văn) đại diện.
3) Danh mục chunks (id + metadata + content rút gọn).

Nhiệm vụ:
- PHÁT HIỆN BẤT THƯỜNG (anomalies) trong chunking, ví dụ:
  * Sai thứ tự/chưa “strict” (nhảy cóc Chương/Điều/Khoản/Điểm).
  * Nhận diện nhầm “Khoản” (không phải dạng `1.`) hoặc “Điểm” (không phải `a)`).
  * Không tiêm intro khoản vào điểm khi đã có chuỗi điểm.
  * Thiếu/bỏ sót nội dung so với excerpts.
  * Metadata không khớp: article_no/clause_no/point_letter/exact_citation.
  * Đóng mở chuỗi điểm sai (bắt đầu không phải “a)”, chèn nội dung thường vào giữa).
  * Nội dung “Điều” không có khoản nhưng không sinh chunk intro.
- GỢI Ý SỬA: chỉ rõ vị trí (id / exact_citation), mô tả vấn đề, cách khắc phục.
- Nếu KHÔNG thấy vấn đề, xác nhận “ok” và nêu ngắn gọn cơ sở kết luận.

Hãy TRẢ LỜI CHỈ ở dạng JSON theo schema:
{
  "status": "ok" | "issues_found",
  "confidence": 0.0-1.0,
  "issues": [
    {
      "id": "chuỗi id chunk hoặc mô tả vị trí",
      "citation": "Điều ... khoản ... điểm ...",
      "severity": "low|medium|high",
      "category": "ordering|regex|metadata|omission|points_chain|format|other",
      "message": "Mô tả ngắn gọn vấn đề",
      "suggestion": "Cách sửa ngắn gọn"
    }
  ],
  "notes": "Ghi chú ngắn (tuỳ chọn)"
}
Trả JSON hợp lệ. Không giải thích ngoài JSON.
"""

def _shorten_text(s: str, max_len: int = 500) -> str:
    if len(s) <= max_len:
        return s
    head = s[: max_len // 2].rstrip()
    tail = s[- max_len // 2 :].lstrip()
    return head + "\n...\n" + tail

def build_review_payload(chunks, summary, raw_text: str, sample_excerpts_chars: int = 8000):
    n = len(raw_text)
    if n <= sample_excerpts_chars:
        excerpts = raw_text
    else:
        k = sample_excerpts_chars // 3
        excerpts = raw_text[:k] + "\n...\n" + raw_text[n//2 - k//2 : n//2 + k//2] + "\n...\n" + raw_text[-k:]

    def lite(c):
        return {
            "id": c.get("metadata", {}).get("exact_citation", ""),
            "metadata": c.get("metadata"),
            "content_preview": _shorten_text(c.get("content", ""), 700)
        }

    chunks_lite = [lite(c) for c in chunks]
    return {
        "summary": summary,
        "excerpts": excerpts,
        "chunks_preview": chunks_lite[:1200]
    }

def _env_truthy(v: str) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "y", "on"}

def call_gemini_review(payload: dict, api_key: str = None) -> dict:
    api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Thiếu GEMINI_API_KEY trong môi trường (.env).")
    try:
        # import at call time
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError(f"Thiếu thư viện google-generativeai: {e}")

    genai.configure(api_key=api_key)
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "response_mime_type": "application/json",
    }
    model = genai.GenerativeModel(GEMINI_MODEL_NAME, generation_config=generation_config)
    prompt_parts = [
        {"role": "user", "parts": [{"text": GEMINI_PROMPT}]},
        {"role": "user", "parts": [{"text": json.dumps(payload, ensure_ascii=False)}]},
    ]
    resp = model.generate_content(prompt_parts)
    raw = getattr(resp, "text", None) or (resp.candidates and resp.candidates[0].content.parts[0].text)
    if not raw:
        raise RuntimeError("Gemini không trả ra nội dung.")
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("JSON không phải object")
        data.setdefault("status", "issues_found")
        data.setdefault("issues", [])
        data.setdefault("confidence", 0.0)
        return data
    except Exception as e:
        return {
            "status": "issues_found",
            "confidence": 0.0,
            "issues": [{
                "id": "PARSER",
                "citation": "",
                "severity": "high",
                "category": "other",
                "message": f"Không parse được JSON từ Gemini: {e}",
                "suggestion": "Chạy lại hoặc giảm excerpts."
            }],
            "notes": raw[:2000]
        }

def advanced_chunk_law_document(text, max_length=600):
    """Chia văn bản luật thành chunks theo logic CHÍNH XÁC"""
    print("   🔍 2-Pass advanced parsing with full validation...")
    lines = normalize_lines(text)
    
    # Regex patterns
    ARTICLE_RE = re.compile(r'^Điều\s+(\d+)\s*[\.:]*\s*(.*)$', re.UNICODE)
    CHAPTER_RE = re.compile(r'^Chương\s+([IVXLCDM]+)\s*(.*)$', re.UNICODE|re.IGNORECASE)
    SECTION_RE = re.compile(r'^Mục\s+(\d+)\s*[:\-]?\s*(.*)$', re.UNICODE|re.IGNORECASE)
    CLAUSE_RE = re.compile(r'^\s*(\d+)\.\s*(.*)$', re.UNICODE)
    POINT_RE = re.compile(r'^\s*([a-zA-ZđĐ])\)\s+(.*)$', re.UNICODE)

    # Heuristic for article-level intro that applies to all clauses
    INTRO_CUE_PAT = re.compile(r'(sau đây|bao gồm|gồm các|quy định như sau)\s*:\s*$', re.IGNORECASE | re.UNICODE)

    def is_intro_text_for_clauses(text_intro: str) -> bool:
        if not text_intro:
            return False
        t = text_intro.strip()
        if t.endswith(':'):
            return True
        if INTRO_CUE_PAT.search(t):
            return True
        return False

    def build_article_header(article_no: int, article_title: str):
        t = (article_title or "").strip()
        return f"Điều {article_no}" + (f" {t}" if t else "")
    
    # Pre-scan
    def prescan(lines):
        chapters_nums, articles_nums = [], []
        chapters_labels = []
        for line in lines:
            if not line: continue
            m_ch = CHAPTER_RE.match(line)
            if m_ch:
                n = roman_to_int(m_ch.group(1))
                if n:
                    chapters_nums.append(n)
                    title = (m_ch.group(2) or "").strip()
                    chapters_labels.append(f"Chương {m_ch.group(1).strip()}" + (f" – {title}" if title else ""))
                continue
            m_art = ARTICLE_RE.match(line)
            if m_art:
                num = int(m_art.group(1))
                articles_nums.append(num)
                continue
        return chapters_nums, articles_nums, chapters_labels
    
    # Flush helpers
    def flush_article_intro(chunks, article_no, article_title, article_intro_buf, chapter, section, stats):
        content = (article_intro_buf or "").strip()
        if not content: return
        title_line = f"Điều {article_no}. {article_title}".strip() if article_title else f"Điều {article_no}"
        chunks.append({
            "content": f"{title_line}\n{content}",
            "title": f"Điều {article_no} - {article_title}" if article_title else f"Điều {article_no}",
            "type": "article_intro",
            "metadata": {
                "chapter": chapter,
                "section": section,
                "article_no": article_no,
                "article_title": article_title,
                "exact_citation": f"Điều {article_no}"
            }
        })
        stats["article_intro"] += 1
    
    def flush_clause(chunks, article_no, article_title, clause_no, content, chapter, section, stats, clause_intro=None):
        content = (content or "").strip()
        if not content:
            return
        art_hdr = build_article_header(article_no, article_title)
        if clause_intro:
            intro = clause_intro.rstrip().rstrip(':') + ':'
            full_content = f"{art_hdr} Khoản {clause_no} {intro}\n{content}"
        else:
            full_content = f"{art_hdr} Khoản {clause_no}. {content}"
        metadata = {
            "chapter": chapter,
            "section": section,
            "article_no": article_no,
            "article_title": article_title,
            "clause_no": clause_no,
            "exact_citation": f"Điều {article_no} khoản {clause_no}"
        }
        if clause_intro:
            metadata["clause_intro"] = clause_intro
        chunks.append({
            "content": full_content,
            "title": f"Điều {article_no}, Khoản {clause_no}",
            "type": "clause",
            "metadata": metadata
        })
        stats["clauses"] += 1
    
    def flush_point(chunks, article_no, article_title, clause_no, letter, content, chapter, section, stats, clause_intro=None):
        content = (content or "").strip()
        if not content:
            return
        letter = letter.lower()
        art_hdr = build_article_header(article_no, article_title)
        if clause_intro:
            intro = clause_intro.rstrip().rstrip(':')
            full_content = f"{art_hdr} Khoản {clause_no} {intro}, điểm {letter})\n{content}"
        else:
            full_content = f"{art_hdr} Khoản {clause_no}, điểm {letter}) {content}"
        metadata = {
            "chapter": chapter,
            "section": section,
            "article_no": article_no,
            "article_title": article_title,
            "clause_no": clause_no,
            "point_letter": letter,
            "exact_citation": f"Điều {article_no} khoản {clause_no} điểm {letter})"
        }
        if clause_intro:
            metadata["clause_intro"] = clause_intro
        chunks.append({
            "content": full_content,
            "title": f"Điều {article_no}, Khoản {clause_no}, Điểm {letter})",
            "type": "point",
            "metadata": metadata
        })
        stats["points"] += 1
    
    print(f"   📄 Processing {len(lines):,} lines...")
    
    # PASS 1: Pre-scan
    print("   🔍 Pass 1: Pre-scanning structure...")
    chapters_nums, articles_nums, chapters_labels = prescan(lines)
    chapters_set, articles_set = set(chapters_nums), set(articles_nums)
    
    print(f"      - Pre-scan found {len(chapters_nums)} chapters: {chapters_nums[:10]}{'...' if len(chapters_nums)>10 else ''}")
    print(f"      - Pre-scan found {len(articles_nums)} articles: {articles_nums[:20]}{'...' if len(articles_nums)>20 else ''}")
    
    # PASS 2: Strict chunking
    print("   🔍 Pass 2: Strict parsing with sequence validation...")
    chunks = []
    stats = {"articles": 0, "article_intro": 0, "clauses": 0, "points": 0}
    warnings = []
    
    chapter_label = None
    section_label = None
    article_no = None
    article_title = ""
    expecting_article_title = False
    article_intro_buf = ""
    article_has_any_chunk = False
    
    clause_no = None
    clause_buf = ""
    clause_intro_current = None
    # Article-level intro to inject into all clauses of the article
    article_clause_intro_current = None
    in_points = False
    point_letter = None
    point_buf = ""
    
    expected_chapter = None
    expected_article = None
    seeking_article = False
    
    def close_clause():
        nonlocal clause_no, clause_buf, in_points, point_letter, point_buf, article_has_any_chunk, clause_intro_current
        if clause_no is None: return
        if in_points and point_letter:
            flush_point(chunks, article_no, article_title, clause_no, point_letter,
                        point_buf, chapter_label, section_label, stats, clause_intro_current)
        elif clause_buf.strip():
            flush_clause(chunks, article_no, article_title, clause_no, clause_buf,
                         chapter_label, section_label, stats, article_clause_intro_current)
        article_has_any_chunk = True
        clause_no, clause_buf, in_points, point_letter, point_buf = None, "", False, None, ""
        clause_intro_current = None
    
    def close_article_if_needed():
        nonlocal article_intro_buf, article_has_any_chunk
        if (not article_has_any_chunk) and article_intro_buf.strip():
            flush_article_intro(chunks, article_no, article_title, article_intro_buf,
                                chapter_label, section_label, stats)
        article_intro_buf = ""
        article_has_any_chunk = False
    
    for ln_idx, line in enumerate(lines, start=1):
        if not line: continue
        
        # Seeking article logic
        if seeking_article:
            m_art_seek = ARTICLE_RE.match(line)
            if m_art_seek:
                a_no = int(m_art_seek.group(1))
                if a_no == expected_article:
                    seeking_article = False
                    close_clause()
                    if article_no is not None:
                        close_article_if_needed()
                    article_no = a_no
                    article_title = (m_art_seek.group(2) or "").strip()
                    stats["articles"] += 1
                    if not article_title:
                        expecting_article_title = True
                    expected_article = a_no + 1
                    clause_no = None; clause_buf = ""; in_points = False; point_letter = None; point_buf = ""
                    clause_intro_current = None
                    article_clause_intro_current = None
                    continue
                else:
                    continue
            m_ch_seek = CHAPTER_RE.match(line)
            if m_ch_seek:
                break
            continue
        
        # Expecting article title on next line
        if expecting_article_title:
            if not (CHAPTER_RE.match(line) or SECTION_RE.match(line) or CLAUSE_RE.match(line) or POINT_RE.match(line) or ARTICLE_RE.match(line)):
                article_title = line; expecting_article_title = False; continue
            else:
                expecting_article_title = False
        
        # CHƯƠNG
        m_ch = CHAPTER_RE.match(line)
        if m_ch:
            close_clause()
            if article_no is not None:
                close_article_if_needed()
            article_no = None
            article_title = ""
            article_intro_buf = ""
            expecting_article_title = False
            article_clause_intro_current = None
            
            roman = m_ch.group(1).strip()
            ch_num = roman_to_int(roman) or 0
            ch_title = (m_ch.group(2) or "").strip()
            lbl = f"Chương {roman}" + (f" – {ch_title}" if ch_title else "")
            
            if expected_chapter is None:
                expected_chapter = ch_num + 1
            else:
                if ch_num == expected_chapter:
                    expected_chapter = ch_num + 1
                elif ch_num > expected_chapter:
                    if expected_chapter not in chapters_set:
                        warnings.append(f"Missing Chapter {expected_chapter} - stopping at {lbl}")
                        break
                    else:
                        warnings.append(f"Skipping {lbl} - waiting for Chapter {expected_chapter}")
                        continue
                else:
                    warnings.append(f"Skipping {lbl} - backward chapter number")
                    continue
            
            chapter_label = lbl
            section_label = None
            continue
        
        # MỤC
        m_sec = SECTION_RE.match(line)
        if m_sec:
            close_clause()
            if article_no is not None:
                close_article_if_needed()
            article_no = None
            article_title = ""
            article_intro_buf = ""
            expecting_article_title = False
            article_clause_intro_current = None
            
            sec_no = m_sec.group(1).strip()
            sec_title = (m_sec.group(2) or "").strip()
            section_label = f"Mục {sec_no}" + (f" – {sec_title}" if sec_title else "")
            continue
        
        # ĐIỀU
        m_art = ARTICLE_RE.match(line)
        if m_art:
            a_no = int(m_art.group(1))
            a_title = (m_art.group(2) or "").strip()
            
            if expected_article is None:
                expected_article = a_no + 1
                close_clause()
                if article_no is not None: close_article_if_needed()
                article_no = a_no; article_title = a_title; stats["articles"] += 1
                if not article_title: expecting_article_title = True
                clause_no = None; clause_buf = ""; in_points = False; point_letter = None; point_buf = ""
                clause_intro_current = None
                article_clause_intro_current = None
                continue
            else:
                if a_no == expected_article:
                    expected_article = a_no + 1
                    close_clause()
                    if article_no is not None: close_article_if_needed()
                    article_no = a_no; article_title = a_title; stats["articles"] += 1
                    if not article_title: expecting_article_title = True
                    clause_no = None; clause_buf = ""; in_points = False; point_letter = None; point_buf = ""
                    clause_intro_current = None
                    article_clause_intro_current = None
                    continue
                elif a_no > expected_article:
                    if expected_article not in articles_set:
                        warnings.append(f"Missing Article {expected_article} - stopping at Article {a_no}")
                        break
                    else:
                        warnings.append(f"Skipping Article {a_no} - waiting for Article {expected_article}")
                        seeking_article = True
                        continue
                else:
                    warnings.append(f"Skipping Article {a_no} - backward article number")
                    continue
        
        # Skip if not in any article
        if article_no is None:
            continue
        
        # KHOẢN
        m_k = CLAUSE_RE.match(line)
        if m_k and m_k.group(1).isdigit():
            if article_intro_buf.strip() and is_intro_text_for_clauses(article_intro_buf):
                # Keep article intro to inject into all clauses
                article_clause_intro_current = article_intro_buf.strip()
                article_intro_buf = ""
            elif article_intro_buf.strip():
                # Flush as a standalone article intro chunk
                flush_article_intro(chunks, article_no, article_title, article_intro_buf,
                                    chapter_label, section_label, stats)
                article_intro_buf = ""; article_has_any_chunk = True
            close_clause()
            clause_no = int(m_k.group(1))
            clause_buf = (m_k.group(2) or "").strip()
            in_points = False; point_letter = None; point_buf = ""
            clause_intro_current = None
            continue
        
        # ĐIỂM
        m_p = POINT_RE.match(line)
        if m_p and clause_no is not None:
            letter = m_p.group(1).lower()
            text = (m_p.group(2) or "").strip()
            
            if not in_points:
                if letter != 'a':
                    clause_buf += ("\\n" if clause_buf else "") + f"{letter}) {text}"
                    continue
                clause_intro_current = clause_buf.strip() if clause_buf.strip() else None
                clause_buf = ""
                in_points = True
                point_letter = letter
                point_buf = text
                continue
            
            if point_letter:
                flush_point(chunks, article_no, article_title, clause_no, point_letter,
                            point_buf, chapter_label, section_label, stats, clause_intro_current)
            in_points = True
            point_letter = letter
            point_buf = text
            continue
        
        # Nội dung kéo dài
        if clause_no is not None:
            if in_points and point_letter:
                point_buf += ("\\n" if point_buf else "") + line
            else:
                clause_buf += ("\\n" if clause_buf else "") + line
        else:
            article_intro_buf += ("\\n" if article_intro_buf else "") + line
    
    # Kết thúc file
    close_clause()
    if article_no is not None:
        close_article_if_needed()
    
    # Filter valid chunks
    final_chunks = []
    for chunk in chunks:
        content = chunk['content'].strip()
        if len(content) > 50:
            final_chunks.append(chunk)
    
    print(f"   📊 PROPER advanced parsing results:")
    print(f"      - Processed {stats['articles']} articles (strict sequence)")
    print(f"      - Created {stats['article_intro']} article intros")
    print(f"      - Created {stats['clauses']} clauses")
    print(f"      - Created {stats['points']} points (with clause intro injection)")
    print(f"      - Final valid chunks: {len(final_chunks)}")
    
    if warnings:
        print(f"   ⚠️  Warnings: {len(warnings)} issues detected")
        for w in warnings[:3]:
            print(f"      - {w}")
    
    return final_chunks

def load_law_documents():
    """Load và chunk văn bản luật"""
    print("📚 Loading law document from LuatHonNhan folder...")
    
    law_file_path = "LuatHonNhan/luat_hon_nhan_va_gia_dinh.docx"
    if os.path.exists(law_file_path):
        print(f"✅ Found law file: {law_file_path}")
        
        # Bước 1: Đọc file docx
        print("\n📖 Step 1: Reading DOCX file...")
        law_text = read_docx(law_file_path)
        
        # Bước 2: Chia thành chunks
        print("\n🔨 Step 2: PROPER Advanced chunking with 2-pass validation...")
        law_chunks = advanced_chunk_law_document(law_text, max_length=600)

        # Bước 2.1: (Tuỳ chọn) Thẩm định bằng Gemini
        if _env_truthy(os.getenv("CHUNK_AI_REVIEW", "false")):
            print("\n🤖 AI Review: Calling Gemini to validate chunking...")
            # Xây dựng summary nhẹ từ chunks (không có cảnh báo nội bộ)
            try:
                chapters_seen = []
                seen_chapter_set = set()
                citations = []
                type_counts_tmp = {"article_intro": 0, "clause": 0, "point": 0}
                seen_articles = set()
                for c in law_chunks:
                    md = c.get("metadata", {})
                    ch = md.get("chapter")
                    if ch and ch not in seen_chapter_set:
                        seen_chapter_set.add(ch)
                        chapters_seen.append(ch)
                    cite = md.get("exact_citation")
                    if cite:
                        citations.append(cite)
                    t = c.get("type")
                    if t in type_counts_tmp:
                        type_counts_tmp[t] += 1
                    a_no = md.get("article_no")
                    if a_no is not None:
                        seen_articles.add(a_no)

                summary = {
                    "chapters_seen": chapters_seen,
                    "articles": len(seen_articles),
                    "article_intro": type_counts_tmp.get("article_intro", 0),
                    "clauses": type_counts_tmp.get("clauses", 0) or type_counts_tmp.get("clause", 0),
                    "points": type_counts_tmp.get("points", 0) or type_counts_tmp.get("point", 0),
                    "citations": citations,
                    "warnings": [],
                    "halted_reason": None,
                    "total_chunks": len(law_chunks)
                }

                excerpts_len = int(os.getenv("CHUNK_REVIEW_EXCERPTS", "8000") or 8000)
                payload = build_review_payload(law_chunks, summary, law_text, sample_excerpts_chars=max(2000, excerpts_len))
                review = call_gemini_review(payload)
                status = review.get("status", "issues_found")
                confidence = review.get("confidence", 0.0)
                issues = review.get("issues", []) or []
                notes = review.get("notes", "")
                print(f"- Gemini status: {status} | confidence: {confidence:.2f}")
                if notes:
                    print(f"- Notes: {notes[:4000]}")
                if issues:
                    print("\n===== Gemini Issues =====")
                    for i, it in enumerate(issues[:20], 1):
                        print(f"{i:02d}. [{it.get('severity','?')}] ({it.get('category','other')}) {it.get('citation') or it.get('id') or ''}")
                        print(f"    - {it.get('message','(no message)')}")
                        if it.get('suggestion'):
                            print(f"    → Suggestion: {it['suggestion']}")

                if _env_truthy(os.getenv("CHUNK_STRICT_OK_ONLY", "false")) and status != "ok":
                    print("❌ CHUNK_STRICT_OK_ONLY is set and Gemini did not return 'ok'. Aborting load.")
                    return []
            except Exception as e:
                print(f"⚠️ Gemini review failed: {e}")
                if _env_truthy(os.getenv("CHUNK_STRICT_OK_ONLY", "false")):
                    return []
        
        # Bước 3: Chuẩn bị dữ liệu cho đánh giá
        print("\n🗂️ Step 3: Preparing data for evaluation...")
        law_docs = []
        for i, chunk in enumerate(law_chunks):
            law_docs.append({
                'id': i,
                'title': chunk['title'],
                'text': chunk['content'],
                'length': len(chunk['content']),
                'type': chunk['type'],
                'metadata': chunk.get('metadata', {})
            })
        
        print(f"   ✅ Processed {len(law_docs)} law document chunks")
        print(f"   📊 Average chunk length: {np.mean([doc['length'] for doc in law_docs]):.0f} characters")
        
        # Thống kê theo loại
        type_counts = {}
        for doc in law_docs:
            doc_type = doc['type']
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        print(f"\n📈 Chunk distribution by type:")
        for chunk_type, count in type_counts.items():
            print(f"   - {chunk_type}: {count} chunks")
        
        print(f"\n✅ Successfully loaded {len(law_docs)} law document chunks!")
        return law_docs
        
    else:
        print(f"❌ File {law_file_path} not found!")
        print(f"   Expected path: {os.path.abspath(law_file_path)}")
        return []

def get_models_to_evaluate():
    """Danh sách các mô hình cần đánh giá"""
    return [
        {
            'name': 'minhquan6203/paraphrase-vietnamese-law',
            'type': 'transformers',
            'description': 'Mô hình Sentence Similarity đã fine tune trên bộ luật pháp Việt Nam',
            'max_length': 512
        },
        {
            'name': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            'type': 'sentence_transformers',
            'description': 'Mô hình cơ sở đa ngôn ngữ (base model)',
            'max_length': 512
        },
        {
            'name': 'namnguyenba2003/Vietnamese_Law_Embedding_finetuned_v3_256dims',
            'type': 'transformers',
            'description': 'Mô hình embedding luật Việt Nam với 256 dimensions',
            'max_length': 512
        },
        {
            'name': 'truro7/vn-law-embedding',
            'type': 'sentence_transformers',
            'description': 'Mô hình embedding luật Việt Nam bằng Sentence Transformers',
            'max_length': 512
        }
    ]

def get_benchmark_queries():
    """Các query từ benchmark"""
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

def search_top_k(query_embedding, doc_embeddings, k=15):
    """Tìm top-k documents tương tự nhất với query"""
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:k]
    top_scores = similarities[top_indices]
    return top_indices, top_scores

def calculate_metrics(scores, threshold_07=0.7, threshold_05=0.5):
    """Tính toán các metrics đánh giá chất lượng retrieval"""
    return {
        "max_score": float(np.max(scores)),
        "avg_top3": float(np.mean(scores[:3])) if len(scores) >= 3 else float(np.mean(scores)),
        "avg_top5": float(np.mean(scores[:5])) if len(scores) >= 5 else float(np.mean(scores)),
        "avg_top10": float(np.mean(scores[:10])) if len(scores) >= 10 else float(np.mean(scores)),
        "avg_all": float(np.mean(scores)),
        "scores_above_07": int(np.sum(scores >= threshold_07)),
        "scores_above_05": int(np.sum(scores >= threshold_05)),
        "min_score": float(np.min(scores))
    }

def display_search_results(query, law_docs, top_indices, top_scores, max_display=5):
    """Hiển thị kết quả search một cách rõ ràng"""
    print(f"📝 Query: {query}")
    
    # Bảo đảm sắp xếp giảm dần theo điểm số
    if not isinstance(top_scores, np.ndarray):
        top_scores = np.array(top_scores)
    if not isinstance(top_indices, np.ndarray):
        top_indices = np.array(top_indices)
    order = np.argsort(top_scores)[::-1]
    top_indices = top_indices[order]
    top_scores = top_scores[order]

    print(f"🎯 Top {min(max_display, len(top_indices))} Results:")
    
    for i in range(min(max_display, len(top_indices))):
        idx = int(top_indices[i])
        score = float(top_scores[i])
        doc = law_docs[idx]
        
        print(f"\n   {i+1}. Score: {score:.4f} | {doc['title']}")
        print(f"      Type: {doc['type']} | Length: {doc['length']} chars")
        print("      Content:")
        print(doc['text'])
        
        if doc.get('metadata') and doc['metadata']:
            metadata_str = ", ".join([f"{k}: {v}" for k, v in doc['metadata'].items()])
            print(f"      Metadata: {metadata_str}")

def evaluate_single_model(model_info, law_docs, queries, top_k=15, show_detailed_results=True, device="cuda"):
    """Đánh giá một mô hình embedding"""
    print(f"\n{'='*80}")
    print(f"🔍 EVALUATING MODEL: {model_info['name']}")
    print(f"📝 Description: {model_info['description']}")
    print(f"🔧 Type: {model_info['type']} | Max Length: {model_info['max_length']} tokens")
    print(f"{'='*80}")
    
    try:
        # Bước 1: Chuẩn bị texts
        doc_texts = [doc['text'] for doc in law_docs]
        print(f"\n📚 Step 1: Prepared {len(doc_texts)} document texts")
        
        # Bước 2: Encode documents
        print(f"\n🔨 Step 2: Encoding documents...")
        if model_info['type'] == 'sentence_transformers':
            doc_embeddings = encode_with_sentence_transformers(
                doc_texts, 
                model_info['name'], 
                batch_size=16,
                device=device
            )
        else:
            doc_embeddings = encode_with_transformers(
                doc_texts, 
                model_info['name'], 
                max_length=model_info['max_length'],
                batch_size=16,
                device=device
            )
        
        print(f"   ✅ Document embeddings shape: {doc_embeddings.shape}")
        
        # Bước 3: Encode queries
        print(f"\n🔍 Step 3: Encoding queries...")
        if model_info['type'] == 'sentence_transformers':
            query_embeddings = encode_with_sentence_transformers(
                queries, 
                model_info['name'], 
                batch_size=16,
                device=device
            )
        else:
            query_embeddings = encode_with_transformers(
                queries, 
                model_info['name'], 
                max_length=model_info['max_length'],
                batch_size=16,
                device=device
            )
        
        print(f"   ✅ Query embeddings shape: {query_embeddings.shape}")
        
        # Bước 4: Evaluate từng query
        print(f"\n📊 Step 4: Evaluating {len(queries)} queries...")
        query_results = []
        all_metrics = []
        
        for i, query in enumerate(queries):
            print(f"\n   🔍 Query {i+1}/{len(queries)}")
            
            # Search top-k documents
            top_indices, top_scores = search_top_k(
                query_embeddings[i], 
                doc_embeddings, 
                k=top_k
            )
            
            # Calculate metrics
            metrics = calculate_metrics(top_scores)
            all_metrics.append(metrics)
            
            # Store results
            query_result = {
                'query': query,
                'query_id': i,
                'top_indices': top_indices.tolist(),
                'top_scores': top_scores.tolist(),
                'metrics': metrics
            }
            query_results.append(query_result)
            
            # Show detailed results for first few queries
            if show_detailed_results and i < 3:
                display_search_results(query, law_docs, top_indices, top_scores, max_display=3)
                print(f"      📈 Metrics: Max={metrics['max_score']:.4f}, Avg_top5={metrics['avg_top5']:.4f}, Above_0.7={metrics['scores_above_07']}")
        
        # Bước 5: Aggregate metrics
        print(f"\n📈 Step 5: Aggregating metrics...")
        
        # Calculate average metrics across all queries
        avg_metrics = {}
        metric_keys = all_metrics[0].keys()
        for key in metric_keys:
            if key.startswith('scores_above'):
                avg_metrics[f"avg_{key}"] = np.mean([m[key] for m in all_metrics])
            else:
                avg_metrics[f"avg_{key}"] = np.mean([m[key] for m in all_metrics])
        
        # Final result
        final_result = {
            'model_name': model_info['name'],
            'model_type': model_info['type'],
            'model_description': model_info['description'],
            'max_length': model_info['max_length'],
            'num_queries': len(queries),
            'num_documents': len(law_docs),
            'top_k': top_k,
            'query_results': query_results,
            'aggregated_metrics': avg_metrics,
            'evaluation_success': True
        }
        
        # Print summary
        print(f"\n✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print(f"   📊 Average Results:")
        print(f"      - Avg Max Score: {avg_metrics['avg_max_score']:.4f}")
        print(f"      - Avg Top-5 Score: {avg_metrics['avg_avg_top5']:.4f}")
        print(f"      - Avg Above 0.7: {avg_metrics['avg_scores_above_07']:.1f}")
        print(f"      - Avg Above 0.5: {avg_metrics['avg_scores_above_05']:.1f}")
        
        return final_result
        
    except Exception as e:
        print(f"\n❌ EVALUATION FAILED: {str(e)}")
        return {
            'model_name': model_info['name'],
            'model_type': model_info['type'],
            'error': str(e),
            'evaluation_success': False
        }

def run_evaluation_all_models(models_to_evaluate, law_docs, benchmark_queries, device="cuda"):
    """Chạy đánh giá cho tất cả các mô hình"""
    print("🚀 Starting evaluation for all models...")
    
    if not law_docs:
        print("❌ Error: No law documents loaded!")
        return []
    
    print(f"✅ Ready to evaluate with:")
    print(f"   📚 Documents: {len(law_docs)} law chunks")
    print(f"   ❓ Queries: {len(benchmark_queries)} benchmark questions")
    print(f"   🤖 Models: {len(models_to_evaluate)} models to test")
    print(f"   🎯 Top-K: 15 results per query")
    
    evaluation_results = []
    successful_evaluations = 0
    failed_evaluations = 0
    
    for i, model_info in enumerate(models_to_evaluate):
        print(f"\n{'🤖 '*20}")
        print(f"🤖 EVALUATING MODEL {i+1}/{len(models_to_evaluate)}")
        print(f"{'🤖 '*20}")
        
        try:
            result = evaluate_single_model(
                model_info=model_info,
                law_docs=law_docs,
                queries=benchmark_queries,
                top_k=15,
                show_detailed_results=(i < 2),  # Show details for first 2 models only
                device=device
            )
            
            if result['evaluation_success']:
                evaluation_results.append(result)
                successful_evaluations += 1
                print(f"✅ Model {i+1} evaluation completed successfully!")
            else:
                print(f"❌ Model {i+1} evaluation failed: {result.get('error', 'Unknown error')}")
                failed_evaluations += 1
            
            # Wait between models
            if i < len(models_to_evaluate) - 1:
                print(f"⏳ Waiting 2 seconds before next model...")
                time.sleep(2)
                
        except Exception as e:
            print(f"❌ Unexpected error evaluating model {i+1}: {str(e)}")
            failed_evaluations += 1
            continue
    
    # Final summary
    print(f"\n{'='*100}")
    print(f"🎉 EVALUATION SUMMARY")
    print(f"{'='*100}")
    print(f"✅ Successful evaluations: {successful_evaluations}")
    print(f"❌ Failed evaluations: {failed_evaluations}")
    print(f"📊 Total models evaluated: {len(evaluation_results)}")
    
    if evaluation_results:
        print(f"\n📈 Quick Performance Preview:")
        # Sắp xếp theo avg_max_score giảm dần
        preview_sorted = sorted(
            evaluation_results,
            key=lambda r: r['aggregated_metrics']['avg_max_score'],
            reverse=True
        )
        for result in preview_sorted:
            metrics = result['aggregated_metrics']
            model_name = result['model_name'].split('/')[-1]
            print(f"   🤖 {model_name}")
            print(f"      Max Score: {metrics['avg_max_score']:.4f} | Top-5: {metrics['avg_avg_top5']:.4f} | Above 0.7: {metrics['avg_scores_above_07']:.1f}")
    
    return evaluation_results

def generate_final_report(evaluation_results, law_docs, benchmark_queries):
    """Tạo báo cáo chi tiết cuối cùng"""
    print("📊 Generating detailed analysis and final report...")
    
    if not evaluation_results:
        print("❌ No evaluation results available!")
        return None
    
    print(f"\n{'='*100}")
    print(f"📋 COMPREHENSIVE EVALUATION REPORT")
    print(f"{'='*100}")
    
    # Sort models by average max score (best first)
    sorted_results = sorted(
        evaluation_results, 
        key=lambda x: x['aggregated_metrics']['avg_max_score'], 
        reverse=True
    )
    
    print(f"📊 DATASET INFORMATION:")
    print(f"   📚 Law Documents: {len(law_docs)} chunks from Luật Hôn nhân và Gia đình")
    print(f"   ❓ Benchmark Queries: {len(benchmark_queries)} questions")
    print(f"   🔍 Evaluation Method: Top-15 retrieval with cosine similarity")
    print(f"   💾 Storage: RAM-based (no vector database)")
    
    print(f"\n🏆 RANKING BY PERFORMANCE:")
    print(f"   Metric: Average Max Score across all queries")
    
    # Create comparison table
    print(f"\n{'Rank':<4} {'Model':<45} {'Max Score':<10} {'Top-5':<8} {'≥0.7':<6} {'≥0.5':<6} {'Type':<12}")
    print(f"{'-'*95}")
    
    for i, result in enumerate(sorted_results):
        metrics = result['aggregated_metrics']
        model_name = result['model_name'].split('/')[-1][:40]
        model_type = result['model_type']
        
        print(f"{i+1:<4} {model_name:<45} {metrics['avg_max_score']:<10.4f} "
              f"{metrics['avg_avg_top5']:<8.4f} {metrics['avg_scores_above_07']:<6.1f} "
              f"{metrics['avg_scores_above_05']:<6.1f} {model_type:<12}")
    
    # Best model analysis
    best_model = sorted_results[0]
    print(f"\n⭐ RECOMMENDED MODEL:")
    print(f"   🥇 {best_model['model_name']}")
    print(f"   📝 {best_model['model_description']}")
    print(f"   🎯 Performance Highlights:")
    best_metrics = best_model['aggregated_metrics']
    print(f"      - Average Max Score: {best_metrics['avg_max_score']:.4f}")
    print(f"      - Average Top-5 Score: {best_metrics['avg_avg_top5']:.4f}")
    print(f"      - Queries with score ≥ 0.7: {best_metrics['avg_scores_above_07']:.1f} per query")
    print(f"      - Queries with score ≥ 0.5: {best_metrics['avg_scores_above_05']:.1f} per query")
    
    # Performance analysis
    print(f"\n📈 PERFORMANCE ANALYSIS:")
    
    # Calculate overall statistics
    all_max_scores = [r['aggregated_metrics']['avg_max_score'] for r in evaluation_results]
    all_top5_scores = [r['aggregated_metrics']['avg_avg_top5'] for r in evaluation_results]
    
    print(f"   📊 Overall Statistics:")
    print(f"      - Best Max Score: {max(all_max_scores):.4f}")
    print(f"      - Worst Max Score: {min(all_max_scores):.4f}")
    print(f"      - Average Max Score: {np.mean(all_max_scores):.4f}")
    print(f"      - Best Top-5 Score: {max(all_top5_scores):.4f}")
    print(f"      - Average Top-5 Score: {np.mean(all_top5_scores):.4f}")
    
    # Model type analysis
    transformers_models = [r for r in evaluation_results if r['model_type'] == 'transformers']
    sentence_transformers_models = [r for r in evaluation_results if r['model_type'] == 'sentence_transformers']
    
    if transformers_models and sentence_transformers_models:
        print(f"\n🔧 MODEL TYPE COMPARISON:")
        
        trans_avg = np.mean([r['aggregated_metrics']['avg_max_score'] for r in transformers_models])
        sent_avg = np.mean([r['aggregated_metrics']['avg_max_score'] for r in sentence_transformers_models])
        
        print(f"   🔨 Transformers models: {len(transformers_models)} models, avg score: {trans_avg:.4f}")
        print(f"   📦 Sentence-transformers: {len(sentence_transformers_models)} models, avg score: {sent_avg:.4f}")
        
        if trans_avg > sent_avg:
            print(f"   ✅ Transformers models perform better on average (+{trans_avg - sent_avg:.4f})")
        else:
            print(f"   ✅ Sentence-transformers models perform better on average (+{sent_avg - trans_avg:.4f})")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   🎯 For Production Deployment:")
    print(f"      - Primary: {best_model['model_name']}")
    print(f"      - Type: {best_model['model_type']}")
    print(f"      - Max Length: {best_model['max_length']} tokens")
    
    if len(sorted_results) > 1:
        second_best = sorted_results[1]
        print(f"   🥈 Alternative Option:")
        print(f"      - {second_best['model_name']}")
        print(f"      - Performance difference: {best_metrics['avg_max_score'] - second_best['aggregated_metrics']['avg_max_score']:.4f}")
    
    print(f"\n🔍 DETAILED QUERY ANALYSIS:")
    print(f"   📝 Sample Query Performance (Best Model):")
    
    # Show performance on first 3 queries for best model
    best_query_results = best_model['query_results'][:3]
    for i, qr in enumerate(best_query_results):
        print(f"\n   Query {i+1}: {qr['query'][:60]}...")
        print(f"      Max Score: {qr['metrics']['max_score']:.4f}")
        print(f"      Top-3 Average: {qr['metrics']['avg_top3']:.4f}")
        print(f"      Results ≥ 0.7: {qr['metrics']['scores_above_07']}")

    # Top queries by total score (best model)
    print(f"\n🔝 TOP 3 QUERIES BY TOTAL SCORE (Best Model)")
    # Compute total score = sum of retrieved top_k scores for each query
    query_totals = []
    for qr in best_model['query_results']:
        total = float(np.sum(qr.get('top_scores', [])))
        query_totals.append({
            'query_id': qr['query_id'],
            'query': qr['query'],
            'total_score': total,
            'top_indices': qr.get('top_indices', []),
            'top_scores': qr.get('top_scores', [])
        })
    # Sort and take top 3
    query_totals.sort(key=lambda x: x['total_score'], reverse=True)
    top3_queries = query_totals[:3]

    # Build detailed objects and print
    top_queries_detailed = []
    for rank, q in enumerate(top3_queries, start=1):
        print(f"\n   {rank}. Total Score: {q['total_score']:.4f}")
        print(f"      Query: {q['query']}")
        # Print top 3 results for this query
        top_n = min(3, len(q['top_indices']))
        answers = []
        for i in range(top_n):
            idx = int(q['top_indices'][i])
            score = float(q['top_scores'][i])
            if 0 <= idx < len(law_docs):
                doc = law_docs[idx]
                md = doc.get('metadata') or {}
                citation = md.get('exact_citation') or ''
                print(f"         - Rank {i+1}: {score:.4f} | {doc.get('title','')}")
                if citation:
                    print(f"           Citation: {citation}")
                # collect for JSON
                answers.append({
                    'rank': i+1,
                    'score': score,
                    'title': doc.get('title', ''),
                    'citation': citation,
                    'type': doc.get('type', ''),
                    'length': doc.get('length', 0),
                    'content': doc.get('text', '')
                })
            else:
                print(f"         - Rank {i+1}: {score:.4f} | [Index {idx} out of range]")
                answers.append({
                    'rank': i+1,
                    'score': score,
                    'title': '',
                    'citation': '',
                    'type': '',
                    'length': 0,
                    'content': '',
                    'note': f'Index {idx} out of range'
                })
        top_queries_detailed.append({
            'rank': rank,
            'query_id': q['query_id'],
            'query': q['query'],
            'total_score': q['total_score'],
            'top_answers': answers
        })

    # Attach into the best model object for JSON output
    try:
        best_model['top_queries_by_total_score'] = top_queries_detailed
    except Exception:
        pass
    
    # Save summary
    evaluation_summary = {
        'best_model': best_model['model_name'],
        'best_score': best_metrics['avg_max_score'],
        'total_models_evaluated': len(evaluation_results),
        'total_queries': len(benchmark_queries),
        'total_documents': len(law_docs),
        'sorted_results': sorted_results,
        # attach preview for convenience
        'top_queries_preview': [
            {
                'query_id': q['query_id'],
                'query': q['query'],
                'total_score': q['total_score'],
                'top3': [
                    {
                        'doc_index': int(q['top_indices'][i]),
                        'score': float(q['top_scores'][i])
                    } for i in range(min(3, len(q['top_indices'])))
                ]
            } for q in top3_queries
        ],
        'top_queries_by_total_score': top_queries_detailed
    }
    
    print(f"\n{'='*100}")
    print(f"✅ EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"📊 Summary saved to evaluation_summary")
    print(f"📋 Full results available in evaluation_results")
    print(f"{'='*100}")
    
    # Export option
    print(f"\n💾 EXPORT OPTIONS:")
    print(f"   To save results to JSON file:")
    print(f"   → import json")
    print(f"   → with open('embedding_evaluation_results.json', 'w', encoding='utf-8') as f:")
    print(f"   →     json.dump(evaluation_results, f, ensure_ascii=False, indent=2)")
    
    print(f"\n🎉 Report generation completed!")
    
    return evaluation_summary

def test_single_query(best_model_name, models_to_evaluate, law_docs, device="cuda"):
    """Test với một query đơn lẻ"""
    print("🧪 Quick test with single query...")
    
    test_query = "Điều kiện kết hôn của nam và nữ theo pháp luật Việt Nam là gì?"
    print(f"📝 Test Query: {test_query}")
    print(f"🥇 Using best model: {best_model_name}")
    
    # Find model info
    best_model_info = None
    for model in models_to_evaluate:
        if model['name'] == best_model_name:
            best_model_info = model
            break
    
    if best_model_info:
        print(f"🔍 Testing single query with model `{best_model_info['name']}`...")
        
        # Encode query
        if best_model_info['type'] == 'sentence_transformers':
            test_query_embedding = encode_with_sentence_transformers([test_query], best_model_info['name'], device=device)[0]
        else:
            test_query_embedding = encode_with_transformers([test_query], best_model_info['name'], best_model_info['max_length'], device=device)[0]
        
        # Encode documents
        doc_texts = [doc['text'] for doc in law_docs]
        if best_model_info['type'] == 'sentence_transformers':
            doc_embeddings = encode_with_sentence_transformers(doc_texts, best_model_info['name'], device=device)
        else:
            doc_embeddings = encode_with_transformers(doc_texts, best_model_info['name'], best_model_info['max_length'], device=device)
        
        # Search
        top_indices, top_scores = search_top_k(test_query_embedding, doc_embeddings, k=10)
        
        # Display results
        display_search_results(test_query, law_docs, top_indices, top_scores, max_display=5)
        
        # Metrics
        metrics = calculate_metrics(top_scores)
        print(f"\n📊 Metrics for this query:")
        print(f"   - Max Score: {metrics['max_score']:.4f}")
        print(f"   - Top-5 Average: {metrics['avg_top5']:.4f}")
        print(f"   - Results ≥ 0.7: {metrics['scores_above_07']}")
        print(f"   - Results ≥ 0.5: {metrics['scores_above_05']}")
    else:
        print("❌ Could not find model info for testing")
    
    print(f"\n✅ Single query test completed!")

def main():
    """Hàm chính chạy toàn bộ evaluation"""
    print("=" * 80)
    print("🔬 ĐÁNH GIÁ MÔ HÌNH EMBEDDING CHO LUẬT TIẾNG VIỆT")
    print("=" * 80)
    
    # 1. Setup environment
    device = setup_environment()
    
    # 2. Load law documents
    law_docs = load_law_documents()
    if not law_docs:
        print("❌ Cannot proceed without law documents!")
        return
    
    # 3. Get models and queries
    models_to_evaluate = get_models_to_evaluate()
    benchmark_queries = get_benchmark_queries()
    
    print(f"\n🤖 Prepared {len(models_to_evaluate)} models for evaluation:")
    for i, model in enumerate(models_to_evaluate):
        print(f"   {i+1}. {model['name']}")
        print(f"      Type: {model['type']} | Max Length: {model['max_length']} tokens")
        print(f"      Description: {model['description']}")
        print()
    
    print(f"🎯 All models support ≥512 tokens as required!")
    print(f"💾 Embeddings will be stored in RAM (not vector database)")
    
    print(f"\nPrepared {len(benchmark_queries)} benchmark queries")
    print("Sample queries:")
    for i, query in enumerate(benchmark_queries[:5]):
        print(f"{i+1}. {query}")
    
    # 4. Run evaluation for all models
    evaluation_results = run_evaluation_all_models(models_to_evaluate, law_docs, benchmark_queries, device)
    
    # 5. Generate final report
    evaluation_summary = generate_final_report(evaluation_results, law_docs, benchmark_queries)
    
    # 6. Test single query with best model
    if evaluation_summary:
        test_single_query(evaluation_summary['best_model'], models_to_evaluate, law_docs, device)
    
    # 7. Save results to JSON
    if evaluation_results:
        try:
            with open('embedding_evaluation_results.json', 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Results saved to: embedding_evaluation_results.json")
        except Exception as e:
            print(f"\n⚠️ Could not save results to JSON: {e}")
    
    print(f"\n🎉 EVALUATION COMPLETED! 🎉")

if __name__ == "__main__":
    main()
