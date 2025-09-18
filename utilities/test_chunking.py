#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json

# Copy các hàm từ embedding_evaluation.py
ROMAN_MAP = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}

def roman_to_int(s: str):
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

def normalize_lines(text: str):
    return [re.sub(r'\s+$', '', ln) for ln in text.splitlines()]

def chunk_law_document(text, law_id="LAW", law_no="", law_title=""):
    """Chia văn bản luật thành chunks theo định dạng hn2014_chunks.json"""
    print("   🔍 Chunking law document with strict parsing...")

    lines = normalize_lines(text)

    # Regex patterns
    ARTICLE_RE = re.compile(r'^Điều\s+(\d+)\s*[\.:]?\s*(.*)$', re.UNICODE)
    CHAPTER_RE = re.compile(r'^Chương\s+([IVXLCDM]+)\s*(.*)$', re.UNICODE | re.IGNORECASE)
    SECTION_RE = re.compile(r'^Mục\s+(\d+)\s*[:\-]?\s*(.*)$', re.UNICODE | re.IGNORECASE)
    CLAUSE_RE = re.compile(r'^\s*(\d+)\.\s*(.*)$', re.UNICODE)
    POINT_RE = re.compile(r'^\s*([a-zA-ZđĐ])\)\s+(.*)$', re.UNICODE)

    # PASS 1: Pre-scan
    chapters_nums, articles_nums = [], []
    chapters_labels, article_titles_seen = [], []

    for line in lines:
        if not line:
            continue
        m_ch = CHAPTER_RE.match(line)
        if m_ch:
            n = roman_to_int(m_ch.group(1))
            if n:
                chapters_nums.append(n)
                title = (m_ch.group(2) or "").strip()
                chapters_labels.append(
                    f"Chương {m_ch.group(1).strip()}" + (f" – {title}" if title else "")
                )
            continue
        m_art = ARTICLE_RE.match(line)
        if m_art:
            articles_nums.append(int(m_art.group(1)))
            continue

    chapters_set, articles_set = set(chapters_nums), set(articles_nums)

    # PASS 2: Chunking
    chunks = []
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
    in_points = False
    point_letter = None
    point_buf = ""
    expected_chapter = None
    expected_article = None
    seeking_article = False

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
            "section": section_label,
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
        content = clause_buf.strip()
        if not content:
            return
        cid = f"{law_id}-D{article_no}-K{clause_no}"
        exact = f"Điều {article_no} khoản {clause_no}"
        meta = {
            "law_no": law_no,
            "law_title": law_title,
            "law_id": law_id,
            "chapter": chapter_label,
            "section": section_label,
            "article_no": article_no,
            "article_title": article_title,
            "clause_no": clause_no,
            "exact_citation": exact
        }
        art_hdr = build_article_header(article_no, article_title)
        full_content = f"{art_hdr} Khoản {clause_no}. {content}"
        chunks.append({
            "id": cid,
            "content": full_content,
            "metadata": meta
        })

    def flush_point():
        nonlocal point_buf
        content = point_buf.strip()
        if not content:
            return
        letter = point_letter.lower()
        cid = f"{law_id}-D{article_no}-K{clause_no}-{letter}"
        exact = f"Điều {article_no} khoản {clause_no} điểm {letter})"
        meta = {
            "law_no": law_no,
            "law_title": law_title,
            "law_id": law_id,
            "chapter": chapter_label,
            "section": section_label,
            "article_no": article_no,
            "article_title": article_title,
            "clause_no": clause_no,
            "point_letter": letter,
            "exact_citation": exact
        }
        if clause_intro_current:
            meta["clause_intro"] = clause_intro_current

        art_hdr = build_article_header(article_no, article_title)
        if clause_intro_current:
            intro = clause_intro_current.rstrip().rstrip(':')
            full_content = f"{art_hdr} Khoản {clause_no} {intro}, điểm {letter}): {content}"
        else:
            full_content = f"{art_hdr} Khoản {clause_no}, điểm {letter}) {content}"

        chunks.append({
            "id": cid,
            "content": full_content,
            "metadata": meta
        })

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

    print(f"   📄 Processing {len(lines):,} lines...")

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
            if not any(regex.match(line) for regex in [CHAPTER_RE, SECTION_RE, CLAUSE_RE, POINT_RE, ARTICLE_RE]):
                article_title = line
                expecting_article_title = False
                continue
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

            roman = m_ch.group(1).strip()
            ch_num = roman_to_int(roman) or 0
            ch_title = (m_ch.group(2) or "").strip()
            lbl = f"Chương {roman}" + (f" – {ch_title}" if ch_title else "")

            if expected_chapter is None:
                expected_chapter = ch_num + 1
            elif ch_num == expected_chapter:
                expected_chapter = ch_num + 1
            elif ch_num > expected_chapter:
                if expected_chapter not in chapters_set:
                    break
                continue
            else:
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
            if article_intro_buf.strip():
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

        # ĐIỂM
        m_p = POINT_RE.match(line)
        if m_p and clause_no is not None:
            letter = m_p.group(1).lower()
            text = (m_p.group(2) or "").strip()

            if not in_points:
                if letter != 'a':
                    clause_buf += ("\n" if clause_buf else "") + f"{letter}) {text}"
                    continue
                clause_intro_current = clause_buf.strip() if clause_buf.strip() else None
                clause_buf = ""
                in_points = True
                point_letter = letter
                point_buf = text
                continue

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

    print(f"   ✅ Created {len(valid_chunks)} chunks")
    return valid_chunks

# Test với một đoạn text mẫu
test_text = """
Điều 1. Phạm vi điều chỉnh

Luật này quy định chế độ hôn nhân và gia đình; chuẩn mực pháp lý cho cách ứng xử giữa các thành viên gia đình; trách nhiệm của cá nhân, tổ chức, Nhà nước và xã hội trong việc xây dựng, củng cố chế độ hôn nhân và gia đình.

Điều 2. Những nguyên tắc cơ bản của chế độ hôn nhân và gia đình

1. Hôn nhân tự nguyện, tiến bộ, một vợ một chồng, vợ chồng bình đẳng.

2. Hôn nhân giữa công dân Việt Nam thuộc các dân tộc, tôn giáo, giữa người theo tôn giáo với người không theo tôn giáo, giữa công dân Việt Nam với người nước ngoài được tôn trọng và được pháp luật bảo vệ.

3. Xây dựng gia đình ấm no, tiến bộ, hạnh phúc; các thành viên gia đình có nghĩa vụ tôn trọng, quan tâm, chăm sóc, giúp đỡ nhau; không phân biệt đối xử giữa các con.

a) Trường hợp này áp dụng khi cha mẹ ly hôn.

b) Trường hợp cha mẹ chết hoặc mất năng lực hành vi dân sự.
"""

if __name__ == "__main__":
    chunks = chunk_law_document(test_text, law_id='TEST', law_no='123/2024/QH15', law_title='Luật Test')

    print(f"\n=== RESULTS ===")
    print(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"ID: {chunk['id']}")
        print(f"Content: {chunk['content'][:300]}...")
        print(f"Metadata: {json.dumps(chunk['metadata'], ensure_ascii=False, indent=2)}")

    # Save to file
    with open('test_chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to test_chunks.json")
