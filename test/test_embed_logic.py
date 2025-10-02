#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script để validate logic embed mà không cần dependencies phức tạp
"""

import os
import json
import argparse

def load_chunks_from_json(json_path: str):
    """Load chunks từ file JSON"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Chunk file not found: {json_path}")

    print(f"📖 Loading chunks from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"   ✅ Loaded {len(chunks)} chunks")

    # Validate chunks có content
    valid_chunks = []
    for chunk in chunks:
        if chunk.get('content', '').strip():
            valid_chunks.append(chunk)
        else:
            print(f"   ⚠️ Skipped chunk with empty content: {chunk.get('id', 'unknown')}")

    print(f"   ✅ Valid chunks: {len(valid_chunks)}")
    return valid_chunks

def extract_texts_from_chunks(chunks):
    """Extract text content từ chunks để embedding"""
    texts = []
    for chunk in chunks:
        content = chunk.get('content', '').strip()
        if content:
            texts.append(content)
    return texts

def create_collection_name(model_name: str, category: str) -> str:
    """Tạo collection name theo format: model_name/category"""
    # Clean model name (remove special chars, replace / with -)
    clean_model = model_name.replace('/', '-').replace('_', '-')
    return f"{clean_model}-{category}"

def test_embed_logic():
    """Test logic embed"""
    parser = argparse.ArgumentParser(description="Test embed logic")
    parser.add_argument("--chunk-file", default="data/BDS_chunk_155638_250925.json")
    parser.add_argument("--model", default="minhquan6203/paraphrase-vietnamese-law")
    parser.add_argument("--category", default="BDS")

    args = parser.parse_args()

    print("=" * 60)
    print("🧪 TESTING EMBED LOGIC")
    print("=" * 60)
    print(f"📁 Chunk file: {args.chunk_file}")
    print(f"🤖 Model: {args.model}")
    print(f"📂 Category: {args.category}")
    print(f"🎯 Collection: {create_collection_name(args.model, args.category)}")
    print("=" * 60)

    try:
        # 1. Load chunks
        chunks = load_chunks_from_json(args.chunk_file)

        # 2. Extract texts
        texts = extract_texts_from_chunks(chunks)
        print(f"📝 Extracted {len(texts)} texts for embedding")

        # Sample texts
        print("\n📄 Sample texts:")
        for i, text in enumerate(texts[:3]):
            print(f"  {i+1}. {text[:100]}...")

        # 3. Show collection name
        collection_name = create_collection_name(args.model, args.category)
        print(f"\n📋 Collection name: {collection_name}")

        # 4. Show sample chunk structure
        if chunks:
            print(f"\n🏗️ Sample chunk structure:")
            sample = chunks[0]
            print(f"  ID: {sample.get('id')}")
            print(f"  Content length: {len(sample.get('content', ''))}")
            print(f"  Metadata keys: {list(sample.get('metadata', {}).keys())}")

        print("\n✅ TEST COMPLETED SUCCESSFULLY!")
        print(f"   Chunks: {len(chunks)}")
        print(f"   Texts: {len(texts)}")
        print(f"   Collection: {collection_name}")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_embed_logic()
