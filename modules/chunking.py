#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vietnamese Law Document Chunking Module
Tách văn bản luật thành chunks theo cấu trúc pháp điển Việt Nam
"""

import re
import os
from typing import List, Dict, Any, Optional

# Roman numeral converter
ROMAN_MAP = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}

def roman_to_int(s: str) -> Optional[int]:
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

# ===== Heuristic intro Điều dành cho các khoản =====
INTRO_CUE_PAT = re.compile(r'(sau đây|bao gồm|gồm các|quy định như sau)\s*:\s*$', re.IGNORECASE | re.UNICODE)
def is_intro_text_for_clauses(text: str) -> bool:
    """
    Heuristic: intro Điều áp dụng cho các khoản nếu:
      - Kết thúc bằng ':' HOẶC
      - Chứa các cụm phổ biến ('sau đây:', 'bao gồm:', 'quy định như sau:'...)
    """
    if not text:
        return False
    t = text.strip()
    if t.endswith(':'):
        return True
    if INTRO_CUE_PAT.search(t):
        return True
    return False

def chapter_to_int(s: str) -> Optional[int]:
    """Chuyển đổi số Chương (La Mã hoặc Ả Rập) sang số nguyên"""
    s = s.strip().upper()
    if not s:
        return None

    # Thử chuyển đổi số La Mã trước
    roman_num = roman_to_int(s)
    if roman_num is not None:
        return roman_num

    # Nếu không phải số La Mã, thử chuyển đổi số Ả Rập
    try:
        arabic_num = int(s)
        return arabic_num
    except ValueError:
        return None

def read_docx(file_path: str) -> str:
    """Đọc file docx và trả về text (hỗ trợ cả .doc và .docx)"""
    safe_path = file_path.encode('ascii', 'ignore').decode('ascii') or file_path
    print(f"   Reading file: {safe_path}")

    # Kiểm tra extension
    file_ext = file_path.lower().split('.')[-1] if '.' in file_path else ''

    try:
        # Thử đọc bằng python-docx trước (cho file .docx thực sự)
        from docx import Document
        doc = Document(file_path)
        text = "\n".join((p.text or "").strip() for p in doc.paragraphs)
        if text and len(text.strip()) > 10:  # Kiểm tra có nội dung thực sự
            print(f"   Successfully read {len(text):,} characters using python-docx")
            return text
        else:
            print(f"   WARNING: python-docx returned minimal content, trying alternatives...")
    except Exception as e1:
        safe_e1 = str(e1).encode('ascii', 'ignore').decode('ascii') or str(e1)
        print(f"   WARNING: python-docx failed: {safe_e1}")

    # Thử dùng docx2txt cho .docx
    try:
        import docx2txt
        text = docx2txt.process(file_path)
        if text and len(text.strip()) > 10:
            print(f"   Successfully read {len(text):,} characters using docx2txt")
            return text
        else:
            print(f"   WARNING: docx2txt returned minimal content")
    except Exception as e2:
        safe_e2 = str(e2).encode('ascii', 'ignore').decode('ascii') or str(e2)
        print(f"   WARNING: docx2txt failed: {safe_e2}")

    # Thử dùng textract cho cả .doc và .docx
    try:
        import textract
        text = textract.process(file_path).decode('utf-8', errors='ignore')
        if text and len(text.strip()) > 10:
            print(f"   Successfully read {len(text):,} characters using textract")
            return text
        else:
            print(f"   WARNING: textract returned minimal content")
    except Exception as e3:
        safe_e3 = str(e3).encode('ascii', 'ignore').decode('ascii') or str(e3)
        print(f"   WARNING: textract failed: {safe_e3}")

    # Thử dùng pypandoc cho cả .doc và .docx
    try:
        import pypandoc
        text = pypandoc.convert_file(file_path, 'plain', extra_args=['--wrap=none'])
        if text and len(text.strip()) > 10:
            print(f"   Successfully read {len(text):,} characters using pypandoc")
            return text
        else:
            print(f"   WARNING: pypandoc returned minimal content")
    except Exception as e4:
        safe_e4 = str(e4).encode('ascii', 'ignore').decode('ascii') or str(e4)
        print(f"   WARNING: pypandoc failed: {safe_e4}")

    # Cuối cùng thử dùng subprocess với antiword (cho .doc)
    if file_ext == 'doc':
        try:
            import subprocess
            result = subprocess.run(['antiword', file_path, '-w', '0'],
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout and len(result.stdout.strip()) > 10:
                print(f"   Successfully read {len(result.stdout):,} characters using antiword")
                return result.stdout
            else:
                print(f"   WARNING: antiword failed or returned minimal content")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e5:
            safe_e5 = str(e5).encode('ascii', 'ignore').decode('ascii') or str(e5)
            print(f"   WARNING: antiword failed: {safe_e5}")

    # Nếu tất cả đều thất bại
    safe_path = file_path.encode('ascii', 'ignore').decode('ascii') or file_path
    print(f"   ERROR: All reading methods failed for {safe_path}")
    return ""

def normalize_lines(text: str) -> List[str]:
    """Normalize lines for more robust header matching:
    - NFC unicode normalization (fix decomposed diacritics like `ề`)
    - replace common no-break spaces with normal spaces
    - strip trailing whitespace and remove BOM
    """
    if text is None:
        return []
    # Normalize unicode to composed form so regex matches decorated characters
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    lines = text.splitlines()
    out: List[str] = []
    for ln in lines:
        if ln is None:
            continue
        # Replace common non-breaking / narrow no-break spaces
        ln = ln.replace('\u00A0', ' ').replace('\u202F', ' ').replace('\u2009', ' ')
        # Remove BOM if present
        ln = ln.replace('\ufeff', '')
        # Trim trailing whitespace
        ln = re.sub(r'\s+$', '', ln)
        out.append(ln)
    print(f"   Normalized {len(out):,} lines with Unicode NFC normalization")
    return out

def generate_law_id_from_name(name: str) -> str:
    """Helper function để tạo law_id từ tên đã chuẩn hóa"""
    # Từ điển mapping cho các loại luật phổ biến
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
        'luat_hon_nhan_gia_dinh_2014': 'LHNVDG',
        'sở hữu trí tuệ': 'LSHTT',
    }

    name_lower = name.lower().replace('_', ' ')

    # Tìm mapping phù hợp
    for key, value in law_mappings.items():
        # Chuẩn hóa key để loại bỏ dấu cho matching
        import unicodedata
        key_normalized = unicodedata.normalize('NFD', key).encode('ascii', 'ignore').decode('ascii').lower()

        # Check exact match
        if key in name or key in name_lower or key_normalized in name:
            return value

        # Check từng từ trong key có trong name không (cho trường hợp có thêm từ khác)
        # Chỉ check nếu key có ít nhất 2 từ để tránh false positive
        key_words = key.split()
        if len(key_words) >= 2:
            key_words_normalized = [unicodedata.normalize('NFD', w).encode('ascii', 'ignore').decode('ascii').lower()
                                   for w in key_words]
            # Check whole word với boundary để tránh substring match
            import re
            name_normalized = name.lower()
            cond1 = all(re.search(r'\b' + re.escape(word) + r'\b', name_normalized) for word in key_words)
            cond2 = all(re.search(r'\b' + re.escape(word) + r'\b', name_normalized) for word in key_words_normalized)
            if cond1 or cond2:
                return value

    # Nếu không tìm thấy, tạo ID từ chữ cái đầu của các từ quan trọng
    words = re.split(r'[_\s]+', name)
    stop_words = {'số', 'và', 'theo', 'phương', 'thức', 'đối', 'tác', 'công', 'tư', 'luật', 'văn', 'bản', 'hợp', 'nhất', 'năm', 'qđ', 'tt', 'bh', 'vbh', 'vbhn', 'vpqh'}

    important_words = []
    for word in words:
        word_lower = word.lower()
        # Bỏ qua số, từ dừng, và từ quá ngắn
        if len(word) <= 1 or word_lower in stop_words or word.isdigit():
            continue
        # Bỏ qua các pattern số như 2020_QH14
        if '_' in word and any(part.isdigit() for part in word.split('_')):
            continue
        important_words.append(word)

    if important_words:
        # Lấy chữ cái đầu của 2-4 từ quan trọng đầu tiên
        first_letters = ''.join(w[0].upper() for w in important_words[:4])
        result = f"L{first_letters}"
        # Giới hạn độ dài
        return result[:8]  # Tối đa 8 ký tự

    # Fallback cuối cùng
    first_letters = ''.join(w[0].upper() for w in words[:3] if len(w) > 1 and not w.isdigit())
    return f"L{first_letters[:6]}"  # Giới hạn 6 ký tự

def generate_law_id(file_name: str) -> str:
    """Tự động sinh law_id từ tên file"""
    # Loại bỏ extension và chuẩn hóa
    name = file_name.replace('.docx', '').replace('.doc', '').strip()

    # Chuẩn hóa tên để matching (thay dấu gạch dưới bằng space)
    name_normalized = name.lower().replace('_', ' ')
    name_lower = name.lower()

    # Xử lý luật sửa đổi, bổ sung
    amendment_keywords = ['sửa đổi', 'bổ sung', 'sửa đổi, bổ sung']
    is_amendment = False
    for keyword in amendment_keywords:
        if keyword in name_normalized:
            is_amendment = True
            break

    if is_amendment:
        # Tìm tên luật gốc trong phần còn lại
        amendment_part = None
        for keyword in amendment_keywords:
            if keyword in name_normalized:
                # Tách phần tên luật gốc (phần sau keyword)
                parts = name_normalized.split(keyword, 1)
                if len(parts) > 1:
                    amendment_part = parts[1].strip()
                    break

        if amendment_part:
            # Tạo ID cho luật gốc từ amendment_part
            base_law_id = generate_law_id_from_name(amendment_part)
            if base_law_id and base_law_id != 'UNKNOWN':
                return f'LSĐBS{base_law_id}'

        # Fallback: tạo ID sửa đổi chung
        return 'LSĐBS'

    # Xử lý các trường hợp đặc biệt
    if 'luật số' in name_lower and 'qh' in name_lower:
        # Luật số XX_YYYY_QHZZ -> LXAYDUNG (luật xây dựng)
        if 'xây dựng' in name_lower:
            return 'LXAYDUNG'
        # Các luật khác có thể thêm mapping

    if 'văn bản hợp nhất' in name_lower:
        # Văn bản hợp nhất Luật XXX -> LDAUTU
        if 'đầu tư' in name_lower:
            return 'LDAUTU'

    # Xử lý các file VBHN (Văn bản hợp nhất) có thể là luật xây dựng
    if name_lower.startswith('vbhn') or 'vbhn' in name_lower:
        # Thường là luật xây dựng
        return 'LXAYDUNG'

    # Xử lý luật số không có từ khóa rõ ràng
    if 'luật số' in name_lower:
        # Có thể là luật xây dựng nếu không match gì khác
        return 'LXAYDUNG'

    # Nếu không phải luật sửa đổi, tạo ID thông thường
    return generate_law_id_from_name(name)

def chunk_law_document(text: str, law_id: str = "LAW", law_no: str = "", law_title: str = "") -> List[Dict[str, Any]]:
    """Chia văn bản luật thành chunks theo định dạng hn2014_chunks.json"""
    print("   Chunking law document with strict parsing...")

    lines = normalize_lines(text)

    # ===== Regex (đầu dòng) =====
    ARTICLE_RE = re.compile(r'^Điều\s+(\d+)\s*[\.:]?\s*(.*)$', re.UNICODE)
    CHAPTER_RE = re.compile(r'^Chương\s+([IVXLCDM]+|\d+)\s*:?\s*(.*)$', re.UNICODE | re.IGNORECASE)
    CLAUSE_RE = re.compile(r'^\s*(\d+)\.\s*(.*)$', re.UNICODE)  # Khoản: PHẢI "1."
    POINT_RE  = re.compile(r'^\s*([a-zA-ZđĐ])[\)\.]\s+(.*)$', re.UNICODE)  # Điểm: "a)" hoặc "a."

    # ===== PASS 1: Pre-scan CHƯƠNG/ĐIỀU (đầu dòng) =====
    chapters_nums, articles_nums = [], []
    chapters_labels, article_titles_seen = [], []
    expecting_chapter_title = False
    roman_current = None

    for raw in lines:
        line = raw
        if not line:
            continue
        if expecting_chapter_title:
            if not (CHAPTER_RE.match(line) or CLAUSE_RE.match(line) or POINT_RE.match(line) or ARTICLE_RE.match(line)):
                ch_title = line.strip()
                lbl = f"Chương {roman_current} – {ch_title}"
                chapters_labels[-1] = lbl  # cập nhật label cuối
                expecting_chapter_title = False
                continue
            else:
                expecting_chapter_title = False
        m_ch = CHAPTER_RE.match(line)
        if m_ch:
            n = roman_to_int(m_ch.group(1))
            if n:
                chapters_nums.append(n)
                title = (m_ch.group(2) or "").strip()
                lbl = f"Chương {m_ch.group(1).strip()}" + (f" – {title}" if title else "")
                chapters_labels.append(lbl)
                if not title:
                    expecting_chapter_title = True
                    roman_current = m_ch.group(1).strip()
                else:
                    expecting_chapter_title = False
            continue
        m_art = ARTICLE_RE.match(line)
        if m_art:
            num = int(m_art.group(1))
            articles_nums.append(num)
            article_titles_seen.append((m_art.group(2) or "").strip())
            continue

    chapters_set, articles_set = set(chapters_nums), set(articles_nums)

    # ===== PASS 2: Chunk strict =====
    chunks = []
    chapter_label: Optional[str] = None
    expecting_chapter_title: bool = False
    roman_current: Optional[str] = None
    chapter_number: Optional[int] = None

    article_no: Optional[int] = None
    article_title: str = ""
    expecting_article_title: bool = False

    article_intro_buf: str = ""
    article_has_any_chunk: bool = False

    clause_no: Optional[int] = None
    clause_buf: str = ""
    # intro của khoản → tiêm vào mọi điểm của khoản
    clause_intro_current: Optional[str] = None
    # intro của điều → tiêm vào mọi khoản của điều
    article_clause_intro_current: Optional[str] = None

    in_points: bool = False
    point_letter: Optional[str] = None
    point_buf: str = ""

    expected_chapter: Optional[int] = None
    expected_article: Optional[int] = None
    seeking_article: bool = False

    def build_article_header(article_no: int, article_title: str) -> str:
        t = (article_title or "").strip()
        return f"Điều {article_no}" + (f" {t}" if t else "")

    def flush_article_intro():
        nonlocal article_intro_buf, article_has_any_chunk
        content = article_intro_buf.strip()
        if not content:
            return
        cid = f"{law_id}-D{article_no}"
        exact = f"Điều {article_no}"
        meta = {
            "law_no": law_no,
            "law_title": law_title,
            "law_id": law_id,
            "chapter": chapter_label,
            "article_no": article_no,
            "article_title": article_title,
            "exact_citation": exact
        }
        title_line = f"Điều {article_no}. {article_title}".strip() if article_title else f"Điều {article_no}"
        chunks.append({
            "id": cid,
            "content": f"{title_line}\n{content}",
            "metadata": meta
        })
        article_intro_buf = ""
        article_has_any_chunk = True

    def flush_clause():
        nonlocal clause_buf
        content = (clause_buf or "").strip()
        if not content:
            return
        cid = f"{law_id}-D{article_no}-K{clause_no}"
        exact = f"Điều {article_no} khoản {clause_no}"
        meta = {
            "law_no": law_no,
            "law_title": law_title,
            "law_id": law_id,
            "chapter": chapter_label,
            "article_no": article_no,
            "article_title": article_title,
            "clause_no": clause_no,
            "exact_citation": exact,
            "chapter_number": chapter_number
        }
        art_hdr = build_article_header(article_no, article_title)

        # TIÊM intro Điều vào khoản (nếu có)
        if article_clause_intro_current:
            intro = article_clause_intro_current.rstrip().rstrip(':') + ':'
            full_content = f"{art_hdr} Khoản {clause_no}. {intro}\n{content}"
            meta["clause_intro"] = article_clause_intro_current
        else:
            full_content = f"{art_hdr} Khoản {clause_no}. {content}"

        chunks.append({"id": cid, "content": full_content, "metadata": meta})

    def flush_point():
        nonlocal point_buf
        content = (point_buf or "").strip()
        if not content:
            return
        if clause_intro_current:
            intro = clause_intro_current.rstrip().rstrip(':') + ':'
            content = f"{intro}\n{content}"
        letter = point_letter.lower()
        cid = f"{law_id}-D{article_no}-K{clause_no}-{letter}"
        exact = f"Điều {article_no} khoản {clause_no} điểm {letter}."
        meta = {
            "law_no": law_no,
            "law_title": law_title,
            "law_id": law_id,
            "chapter": chapter_label,
            "article_no": article_no,
            "article_title": article_title,
            "clause_no": clause_no,
            "point_letter": letter,
            "exact_citation": exact,
            "chapter_number": chapter_number
        }
        if clause_intro_current:
            meta["clause_intro"] = clause_intro_current
        art_hdr = build_article_header(article_no, article_title)

        if clause_intro_current:
            intro = clause_intro_current.rstrip().rstrip(':')
            full_content = f"{art_hdr} Khoản {clause_no}. {intro}, điểm {letter}.\n{content}"
        else:
            full_content = f"{art_hdr} Khoản {clause_no}, điểm {letter}. {content}"

        chunks.append({"id": cid, "content": full_content, "metadata": meta})

    def close_clause():
        nonlocal clause_no, clause_buf, in_points, point_letter, point_buf, article_has_any_chunk, clause_intro_current
        if clause_no is None:
            return
        if in_points and point_letter:
            flush_point()
        elif clause_buf.strip():
            flush_clause()
        article_has_any_chunk = True
        clause_no, clause_buf, in_points, point_letter, point_buf = None, "", False, None, ""
        clause_intro_current = None

    def close_article_if_needed():
        nonlocal article_intro_buf, article_has_any_chunk
        if not article_has_any_chunk and article_intro_buf.strip():
            flush_article_intro()

    print(f"   Processing {len(lines):,} lines...")

    for line in lines:
        if not line:
            continue

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
                    if not article_title:
                        expecting_article_title = True
                    expected_article = a_no + 1
                    clause_no = None
                    clause_buf = ""
                    in_points = False
                    point_letter = None
                    point_buf = ""
                    clause_intro_current = None
                    continue
            continue

        if expecting_article_title:
            if not any(regex.match(line) for regex in [CHAPTER_RE, CLAUSE_RE, POINT_RE, ARTICLE_RE]):
                article_title = line
                expecting_article_title = False
                continue
            else:
                expecting_article_title = False

        if expecting_chapter_title:
            if not (CHAPTER_RE.match(line) or CLAUSE_RE.match(line) or POINT_RE.match(line) or ARTICLE_RE.match(line)):
                ch_title = line.strip()
                lbl = f"Chương {roman_current} – {ch_title}"
                chapter_label = lbl
                expecting_chapter_title = False
                continue
            else:
                expecting_chapter_title = False

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

            if not ch_title:
                expecting_chapter_title = True
                roman_current = roman  # lưu lại để dùng sau
                chapter_number = ch_num  # lưu chapter_number
            else:
                chapter_label = lbl
                chapter_number = ch_num  # lưu chapter_number
                expecting_chapter_title = False

            if expected_chapter is None:
                expected_chapter = ch_num + 1
            else:
                if ch_num == expected_chapter:
                    expected_chapter = ch_num + 1
                elif ch_num > expected_chapter:
                    if expected_chapter not in chapters_set:
                        break
                    continue
                else:
                    continue

            continue

        # ĐIỀU
        m_art = ARTICLE_RE.match(line)
        if m_art:
            a_no = int(m_art.group(1))
            a_title = (m_art.group(2) or "").strip()

            if expected_article is None:
                expected_article = a_no + 1
                close_clause()
                if article_no is not None:
                    close_article_if_needed()
                article_no = a_no
                article_title = a_title
                if not article_title:
                    expecting_article_title = True
                clause_no = None
                clause_buf = ""
                in_points = False
                point_letter = None
                point_buf = ""
                clause_intro_current = None
                article_clause_intro_current = None
                continue
            elif a_no == expected_article:
                expected_article = a_no + 1
                close_clause()
                if article_no is not None:
                    close_article_if_needed()
                article_no = a_no
                article_title = a_title
                if not article_title:
                    expecting_article_title = True
                clause_no = None
                clause_buf = ""
                in_points = False
                point_letter = None
                point_buf = ""
                clause_intro_current = None
                article_clause_intro_current = None
                continue
            elif a_no > expected_article:
                if expected_article not in articles_set:
                    break
                else:
                    seeking_article = True
                    continue
            else:
                continue

        if article_no is None:
            continue

        # KHOẢN
        m_k = CLAUSE_RE.match(line)
        if m_k and m_k.group(1).isdigit():
            # Nếu Điều có intro 'giới thiệu' → lưu lại để tiêm cho mọi khoản, KHÔNG flush chunk intro Điều
            if article_intro_buf.strip() and is_intro_text_for_clauses(article_intro_buf):
                article_clause_intro_current = article_intro_buf.strip()
                article_intro_buf = ""
            elif article_intro_buf.strip():
                # Intro không phải dạng 'giới thiệu' → tạo chunk Điều intro như cũ
                flush_article_intro()
                article_has_any_chunk = True

            close_clause()
            clause_no = int(m_k.group(1))
            clause_buf = (m_k.group(2) or "").strip()
            in_points = False
            point_letter = None
            point_buf = ""
            clause_intro_current = None
            continue

        # ĐIỂM — chuỗi điểm chỉ bắt đầu nếu mở bằng a)
        m_p = POINT_RE.match(line)
        if m_p:
            # Nếu chưa có khoản, coi như khoản 1 ẩn cho điểm trực tiếp
            if clause_no is None:
                clause_no = 1
                clause_buf = ""
                clause_intro_current = None

            letter = m_p.group(1).lower()
            text = (m_p.group(2) or "").strip()

            if not in_points:
                if letter != 'a':
                    # không coi là điểm → nhập vào nội dung khoản
                    clause_buf += ("\n" if clause_buf else "") + f"{letter}. {text}"
                    continue
                # bắt đầu chuỗi điểm với 'a)' — KHÔNG flush chunk "Khoản X"
                # lưu intro khoản để tiêm vào MỌI điểm
                clause_intro_current = clause_buf.strip() if clause_buf.strip() else None
                clause_buf = ""
                in_points = True
                point_letter = letter
                point_buf = text
                continue

            # đang trong chuỗi điểm: flush điểm trước, mở điểm mới
            if point_letter:
                flush_point()
            in_points = True
            point_letter = letter
            point_buf = text
            continue

        # Nội dung kéo dài
        if clause_no is not None:
            if in_points and point_letter:
                point_buf += ("\n" if point_buf else "") + line
            else:
                clause_buf += ("\n" if clause_buf else "") + line
        else:
            article_intro_buf += ("\n" if article_intro_buf else "") + line

    # Kết thúc
    close_clause()
    if article_no is not None:
        close_article_if_needed()

    # Filter chunks
    valid_chunks = []
    for chunk in chunks:
        content = chunk['content'].strip()
        if len(content) > 50:
            valid_chunks.append(chunk)

    print(f"   Created {len(valid_chunks)} chunks")
    return valid_chunks
