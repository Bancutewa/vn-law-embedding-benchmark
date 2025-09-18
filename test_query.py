#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script đơn giản để hỏi query với Qdrant
"""

import json
from dotenv import load_dotenv
load_dotenv()

# Optional imports
try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

def get_qdrant_client():
    """Khởi tạo Qdrant client từ biến môi trường."""
    if not QDRANT_AVAILABLE:
        print("❌ Qdrant client không khả dụng")
        return None

    try:
        import os

        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        client = QdrantClient(
            url=url,
            api_key=api_key or None,
            timeout=300.0,
            grpc_port=6334,
        )
        print("🗄️ Qdrant connected successfully")
        return client
    except Exception as e:
        print(f"❌ Không thể kết nối Qdrant: {e}")
        return None

def search_qdrant(client, collection_name, query_vector, top_k=5):
    """Tìm kiếm trong Qdrant"""
    try:
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )

        results = []
        for hit in search_result:
            results.append({
                'id': hit.id,
                'score': hit.score,
                'payload': hit.payload
            })

        return results

    except Exception as e:
        print(f"❌ Lỗi tìm kiếm Qdrant: {e}")
        return []

def simple_query_test():
    """Test query đơn giản"""

    print("🔍 QUERY SEARCH TOOL")
    print("=" * 30)

    # Kết nối Qdrant
    client = get_qdrant_client()
    if not client:
        print("Không thể kết nối Qdrant. Hãy đảm bảo Qdrant đang chạy.")
        return

    # Chọn model/collection
    collection_name = input("Nhập tên collection (model name, ví dụ: minhquan6203_paraphrase-vietnamese-law): ").strip()
    if not collection_name:
        collection_name = "minhquan6203_paraphrase-vietnamese-law"

    print(f"📚 Sử dụng collection: {collection_name}")

    # Kiểm tra collection tồn tại
    try:
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        if collection_name not in collection_names:
            print(f"⚠️ Collection '{collection_name}' không tồn tại. Các collection có sẵn:")
            for name in collection_names:
                print(f"  - {name}")
            return
    except Exception as e:
        print(f"❌ Lỗi kiểm tra collection: {e}")
        return

    # Nhập query
    query_text = input("\\n📝 Nhập câu hỏi của bạn: ").strip()
    if not query_text:
        query_text = "Điều kiện kết hôn theo pháp luật Việt Nam?"

    print(f"\\n🔍 Đang tìm: {query_text}")

    # Encode query (giả sử dùng SentenceTransformer)
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("❌ Sentence Transformers không khả dụng")
        return

    try:
        # Infer model name from collection name
        model_name = collection_name.replace('_', '/')
        if 'minhquan6203' in model_name:
            model_name = 'minhquan6203/paraphrase-vietnamese-law'
        elif 'namnguyenba2003' in model_name:
            model_name = 'namnguyenba2003/Vietnamese_Law_Embedding_finetuned_v3_256dims'
        elif 'truro7' in model_name:
            model_name = 'truro7/vn-law-embedding'
        else:
            model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

        print(f"🤖 Sử dụng model: {model_name}")

        model = SentenceTransformer(model_name)
        query_vector = model.encode([query_text])[0]

        print("✅ Đã encode query")

    except Exception as e:
        print(f"❌ Lỗi encode query: {e}")
        return

    # Tìm kiếm
    print("🔄 Đang tìm kiếm...")
    results = search_qdrant(client, collection_name, query_vector.tolist(), top_k=5)

    if not results:
        print("❌ Không tìm thấy kết quả")
        return

    print(f"\\n🎯 Tìm thấy {len(results)} kết quả:")

    for i, result in enumerate(results, 1):
        print(f"\\n{i}. Score: {result['score']:.4f}")

        payload = result['payload']
        if payload:
            # Hiển thị thông tin từ payload
            content = payload.get('content', '')[:200] + '...' if len(payload.get('content', '')) > 200 else payload.get('content', '')
            law_id = payload.get('metadata', {}).get('law_id', 'N/A')
            exact_citation = payload.get('metadata', {}).get('exact_citation', 'N/A')

            print(f"   📄 Nội dung: {content}")
            print(f"   🏷️  Law ID: {law_id}")
            print(f"   📍 Trích dẫn: {exact_citation}")
        else:
            print(f"   ID: {result['id']}")

def main():
    """Main function"""
    try:
        while True:
            simple_query_test()
            print("\\n" + "="*50)
            again = input("Tìm kiếm tiếp? (y/n): ").strip().lower()
            if again != 'y':
                break
    except KeyboardInterrupt:
        print("\\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    main()
