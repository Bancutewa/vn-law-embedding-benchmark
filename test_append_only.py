#!/usr/bin/env python3
"""
Test append mode logic only
"""

import os
import sys

# Mock qdrant client
class MockQdrantClient:
    def __init__(self):
        self.collections = {}

    def get_collection(self, collection_name):
        if collection_name in self.collections:
            return self.collections[collection_name]
        return None

    def recreate_collection(self, collection_name, vectors_config):
        # Mock CollectionConfig object
        class MockCollectionConfig:
            def __init__(self, vector_size):
                self.params = type('MockParams', (), {
                    'vectors': type('MockVectors', (), {'size': vector_size})()
                })()

        self.collections[collection_name] = type('MockCollectionInfo', (), {
            'vectors_count': 0,
            'points_count': 0,
            'status': 'green',
            'config': MockCollectionConfig(vectors_config.size)
        })()

# Mock get_collection_info
def mock_get_collection_info(client, collection_name):
    collection = client.get_collection(collection_name)
    if collection:
        return {
            'name': collection_name,
            'vectors_count': collection.vectors_count,
            'points_count': collection.points_count,
            'status': collection.status,
            'config': collection.config
        }
    return None

# Mock ensure_collection
def mock_ensure_collection(client, collection_name, vector_size):
    client.recreate_collection(collection_name, type('MockVectorsConfig', (), {'size': vector_size})())
    print(f"Mock: Ensured collection {collection_name}")

# Import and override
sys.path.append(os.path.dirname(__file__))
import modules.qdrant_manager as qm

qm.get_collection_info = mock_get_collection_info
qm.ensure_collection = mock_ensure_collection

def test_append_logic():
    print("Testing append logic...")

    client = MockQdrantClient()

    # Test 1: Collection không tồn tại
    print("\n1. Collection khong ton tai, append=True")
    try:
        result = qm.ensure_or_append_collection(client, "test_collection", 768, append_mode=True)
        print(f"   Result: {result} (True = created new)")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 2: Collection đã tồn tại, append mode
    print("\n2. Collection da ton tai, append=True")
    try:
        result = qm.ensure_or_append_collection(client, "test_collection", 768, append_mode=True)
        print(f"   Result: {result} (False = append mode)")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 3: Vector size mismatch
    print("\n3. Vector size mismatch")
    try:
        result = qm.ensure_or_append_collection(client, "test_collection", 512, append_mode=True)
        print(f"   Result: {result}")
    except ValueError as e:
        print(f"   Expected error: {e}")

    print("\nAppend logic test completed!")

if __name__ == "__main__":
    test_append_logic()
