#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility: Create/Update Qdrant payload indexes for existing collections.

Usage:
  - Set env QDRANT_URL and QDRANT_API_KEY (optional if not required by server)
  - Run:
      python utilities/update_qdrant_indexes.py --collection minhquan6203_paraphrase-vietnamese-law
    or multiple:
      python utilities/update_qdrant_indexes.py --collection c1 --collection c2
"""

import os
import sys
import argparse

from dotenv import load_dotenv
load_dotenv()

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import PayloadSchemaType
except Exception as e:
    print(f"❌ Missing qdrant_client: {e}")
    sys.exit(1)


INDEX_FIELDS = {
    # Cấu trúc pháp điển
    "metadata.law_id": PayloadSchemaType.KEYWORD,
    "metadata.law_title": PayloadSchemaType.KEYWORD,
    "metadata.law_no": PayloadSchemaType.KEYWORD,
    "metadata.chapter": PayloadSchemaType.KEYWORD,
    "metadata.section": PayloadSchemaType.KEYWORD,
    "metadata.article_no": PayloadSchemaType.INTEGER,
    "metadata.article_title": PayloadSchemaType.KEYWORD,
    "metadata.clause_no": PayloadSchemaType.INTEGER,
    "metadata.point_letter": PayloadSchemaType.KEYWORD,
    "metadata.exact_citation": PayloadSchemaType.KEYWORD,
    # Nguồn
    "metadata.source_category": PayloadSchemaType.KEYWORD,
    "metadata.source_file_name": PayloadSchemaType.KEYWORD,
    "metadata.source_file": PayloadSchemaType.KEYWORD,
    "metadata.chunk_index": PayloadSchemaType.INTEGER,
}


def get_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY") or None
    if not url:
        # Fallback to localhost if not provided
        url = "http://localhost:6333"
    client = QdrantClient(url=url, api_key=api_key, timeout=300.0, grpc_port=6334)
    return client


def ensure_indexes(client: QdrantClient, collection_name: str):
    print(f"\n🔧 Creating indexes on collection: {collection_name}")
    # Validate collection exists
    cols = client.get_collections().collections
    names = {c.name for c in cols}
    if collection_name not in names:
        print(f"❌ Collection '{collection_name}' not found. Available: {sorted(names)}")
        return

    created = 0
    skipped = 0
    for field_name, schema in INDEX_FIELDS.items():
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema,
            )
            print(f"   ✅ Indexed: {field_name} ({schema})")
            created += 1
        except Exception as e:
            # Likely already exists or incompatible; report and continue
            print(f"   ⚠️  Skip '{field_name}': {e}")
            skipped += 1

    print(f"🔎 Done. Created={created}, Skipped={skipped}")


def parse_args():
    p = argparse.ArgumentParser(description="Update Qdrant payload indexes")
    p.add_argument(
        "--collection",
        action="append",
        help="Collection name (can be specified multiple times)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    collections = args.collection or []
    if not collections:
        # Suggested defaults matching evaluated models
        collections = [
            "minhquan6203_paraphrase-vietnamese-law",
            "sentence-transformers_paraphrase-multilingual-mpnet-base-v2",
            "namnguyenba2003_Vietnamese_Law_Embedding_finetuned_v3_256dims",
            "truro7_vn-law-embedding",
        ]
        print("⚠️ No --collection provided. Will try common defaults:")
        for c in collections:
            print(f"   - {c}")

    try:
        client = get_client()
    except Exception as e:
        print(f"❌ Cannot connect to Qdrant: {e}")
        sys.exit(1)

    for col in collections:
        ensure_indexes(client, col)

    print("\n✅ Index update finished.")


if __name__ == "__main__":
    main()


