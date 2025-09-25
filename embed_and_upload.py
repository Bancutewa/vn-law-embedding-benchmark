#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embed chunks và upload lên Qdrant Database
Tái sử dụng được cho các category và model khác nhau
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Import từ modules
from modules.qdrant_manager import (
    get_qdrant_client,
    ensure_collection,
    upsert_embeddings_to_qdrant,
    count_collection_points,
    get_collection_info
)
from modules.embedding_models import encode_texts, get_embedding_dimension

def load_chunks_from_json(json_path: str) -> List[Dict[str, Any]]:
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

def extract_texts_from_chunks(chunks: List[Dict[str, Any]]) -> List[str]:
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

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Embed chunks và upload lên Qdrant Database",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ========== THAM SỐ CẦN SỬA ==========
    # File chunk JSON - sửa path này khi cần
    parser.add_argument(
        "--chunk-file",
        default="data/BDS_chunk_155638_250925.json",  # ←←← SỬA PATH FILE CHUNK Ở ĐÂY
        help="Path to chunk JSON file (default: data/BDS_chunk_155638_250925.json)"
    )

    # Model embedding - sửa model này khi cần
    parser.add_argument(
        "--model",
        default="minhquan6203/paraphrase-vietnamese-law",  # ←←← SỬA MODEL Ở ĐÂY
        help="Embedding model name (default: minhquan6203/paraphrase-vietnamese-law)"
    )

    # Category - sửa category này khi cần
    parser.add_argument(
        "--category",
        default="BDS",  # ←←← SỬA CATEGORY Ở ĐÂY
        help="Category name for collection (default: BDS)"
    )

    # Model type - tự động detect nhưng có thể override
    parser.add_argument(
        "--model-type",
        choices=["transformers", "sentence_transformers"],
        help="Force model type (auto-detected if not specified)"
    )

    # Các tham số khác
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding (default: 16)"
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for embedding (default: cuda)"
    )

    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreate collection even if it exists"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test mode: chỉ encode và validate, không upload lên Qdrant"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 VIETNAMESE LAW EMBEDDING & QDRANT UPLOAD")
    print("=" * 80)
    print(f"📁 Chunk file: {args.chunk_file}")
    print(f"🤖 Model: {args.model}")
    print(f"📂 Category: {args.category}")
    print(f"🎯 Collection: {create_collection_name(args.model, args.category)}")
    print(f"⚙️  Device: {args.device}")
    print(f"📦 Batch size: {args.batch_size}")
    if args.dry_run:
        print("🔍 DRY RUN MODE - No upload to Qdrant")
    print("=" * 80)

    try:
        # 1. Load chunks từ JSON
        chunks = load_chunks_from_json(args.chunk_file)

        if not chunks:
            print("❌ No valid chunks found!")
            return

        # 2. Extract texts để embedding
        texts = extract_texts_from_chunks(chunks)
        print(f"📝 Extracted {len(texts)} texts for embedding")

        # 3. Setup model info
        # Auto-detect model type
        model_type = args.model_type
        if not model_type:
            # Auto-detect based on model name
            if 'sentence-transformers' in args.model or 'all-MiniLM' in args.model or 'paraphrase-multilingual' in args.model:
                model_type = 'sentence_transformers'
            else:
                model_type = 'transformers'  # Default for most HF models

        model_info = {
            'name': args.model,
            'type': model_type,
            'max_length': 512
        }

        print(f"🎯 Detected model type: {model_type}")

        # 4. Get embedding dimension
        print(f"📏 Getting embedding dimension...")
        vector_size = get_embedding_dimension(model_info)
        print(f"   ✅ Embedding dimension: {vector_size}")

        if args.dry_run:
            print("🔍 DRY RUN: Skipping embedding and upload")
            return

        # 5. Encode texts thành embeddings
        print(f"🧠 Encoding {len(texts)} texts with {args.model}...")
        embeddings = encode_texts(
            texts=texts,
            model_info=model_info,
            device=args.device
        )

        print(f"   ✅ Generated embeddings shape: {embeddings.shape}")

        # 6. Kết nối Qdrant
        print(f"🗄️  Connecting to Qdrant...")
        client = get_qdrant_client()

        # 7. Tạo collection name
        collection_name = create_collection_name(args.model, args.category)
        print(f"📋 Collection name: {collection_name}")

        # 8. Check existing collection
        existing_info = get_collection_info(client, collection_name)
        if existing_info and not args.force_recreate:
            existing_count = existing_info.get('points_count', 0)
            print(f"⚠️  Collection '{collection_name}' already exists with {existing_count} points")
            choice = input("   Continue and recreate? (y/N): ").strip().lower()
            if choice != 'y':
                print("❌ Aborted")
                return

        # 9. Tạo collection
        print(f"🏗️  Creating collection '{collection_name}'...")
        ensure_collection(client, collection_name, vector_size)

        # 10. Upload embeddings và chunks lên Qdrant
        print(f"📤 Uploading to Qdrant...")
        upsert_embeddings_to_qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings,
            law_docs=chunks,
            batch_size=100  # Qdrant batch size
        )

        # 11. Verify upload
        final_count = count_collection_points(client, collection_name)
        print(f"✅ SUCCESS! Collection '{collection_name}' now has {final_count} vectors")

        print("\n🎉 EMBEDDING & UPLOAD COMPLETED!")
        print(f"   Collection: {collection_name}")
        print(f"   Vectors: {final_count}")
        print(f"   Dimension: {vector_size}")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
