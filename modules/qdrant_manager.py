#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qdrant Vector Database Manager
Quản lý các operations với Qdrant vector database
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

def get_qdrant_client() -> Optional[QdrantClient]:
    """Khởi tạo Qdrant client từ biến môi trường."""
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    try:
        client = QdrantClient(
            url=url,
            api_key=api_key or None,
            timeout=300.0,
            grpc_port=6334,
        )
        print("🗄️  Qdrant connected successfully")
        return client
    except Exception as e:
        print(f"❌ Cannot connect to Qdrant: {e}")
        raise

def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """Tạo mới (recreate) collection với cấu hình cosine và kích thước vector tương ứng."""
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"✅ Collection ready: {collection_name} (dim={vector_size})")

    try:
        from qdrant_client.http.models import PayloadSchemaType

        index_fields = {
            # Cấu trúc pháp điển
            "metadata.law_id": PayloadSchemaType.keyword,
            "metadata.law_title": PayloadSchemaType.keyword,
            "metadata.law_no": PayloadSchemaType.keyword,
            "metadata.chapter": PayloadSchemaType.keyword,
            "metadata.section": PayloadSchemaType.keyword,
            "metadata.article_no": PayloadSchemaType.integer,
            "metadata.article_title": PayloadSchemaType.keyword,
            "metadata.clause_no": PayloadSchemaType.integer,
            "metadata.point_letter": PayloadSchemaType.keyword,
            "metadata.exact_citation": PayloadSchemaType.keyword,
            # Nguồn
            "metadata.source_category": PayloadSchemaType.keyword,
            "metadata.source_file_name": PayloadSchemaType.keyword,
            "metadata.source_file": PayloadSchemaType.keyword,
            "metadata.chunk_index": PayloadSchemaType.integer,
        }

        for field_name, schema_type in index_fields.items():
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
                print(f"   🔎 Indexed payload field: {field_name} ({schema_type})")
            except Exception as ie:
                # Có thể index đã tồn tại sau khi recreate; chỉ log cảnh báo nhẹ
                print(f"   ⚠️ Could not index '{field_name}': {ie}")
    except Exception as e:
        print(f"⚠️ Skipped creating payload indexes: {e}")

def upsert_embeddings_to_qdrant(client: QdrantClient, collection_name: str, embeddings: np.ndarray, law_docs: List[Dict[str, Any]], batch_size: int = 100) -> None:
    """Upsert toàn bộ embeddings và payload vào Qdrant theo batch nhỏ."""
    total_points = len(embeddings)
    print(f"📤 Upserting {total_points} vectors in batches of {batch_size}...")

    for i in range(0, total_points, batch_size):
        batch_end = min(i + batch_size, total_points)
        batch_points = []

        for idx in range(i, batch_end):
            payload = law_docs[idx]
            batch_points.append(PointStruct(id=idx, vector=embeddings[idx].tolist(), payload=payload))

        # Retry mechanism
        max_retries = 3
        for retry in range(max_retries):
            try:
                client.upsert(collection_name=collection_name, points=batch_points)
                print(f"   ✅ Batch {i//batch_size + 1}: upserted {len(batch_points)} vectors ({batch_end}/{total_points})")
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"   ⚠️ Batch {i//batch_size + 1} failed (attempt {retry + 1}/{max_retries}): {e}")
                    print(f"   🔄 Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    print(f"   ❌ Batch {i//batch_size + 1} failed after {max_retries} attempts: {e}")
                    raise

    print(f"✅ Successfully upserted {total_points} vectors into Qdrant collection '{collection_name}'")

def search_qdrant(client: QdrantClient, collection_name: str, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Search top-k bằng Qdrant, trả về (indices, scores)."""
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=top_k
    )
    top_indices = [hit.id for hit in hits]
    top_scores = [float(hit.score) for hit in hits]
    return np.array(top_indices), np.array(top_scores, dtype=float)

def count_collection_points(client: QdrantClient, collection_name: str) -> int:
    """Đếm số điểm trong collection"""
    try:
        ct = client.count(collection_name=collection_name, exact=True)
        return int(getattr(ct, 'count', 0))
    except Exception:
        return 0

def delete_collection(client: QdrantClient, collection_name: str) -> None:
    """Xóa collection"""
    try:
        client.delete_collection(collection_name=collection_name)
        print(f"🗑️  Deleted collection: {collection_name}")
    except Exception as e:
        print(f"⚠️ Could not delete collection '{collection_name}': {e}")

def list_collections(client: QdrantClient) -> List[str]:
    """Liệt kê tất cả collections"""
    try:
        collections = client.get_collections()
        return [col.name for col in collections.collections]
    except Exception as e:
        print(f"❌ Error listing collections: {e}")
        return []

def get_collection_info(client: QdrantClient, collection_name: str) -> Optional[Dict[str, Any]]:
    """Lấy thông tin về collection"""
    try:
        info = client.get_collection(collection_name=collection_name)
        return {
            'name': collection_name,
            'vectors_count': info.vectors_count,
            'points_count': info.points_count,
            'status': info.status,
            'config': info.config
        }
    except Exception as e:
        print(f"❌ Error getting collection info for '{collection_name}': {e}")
        return None
