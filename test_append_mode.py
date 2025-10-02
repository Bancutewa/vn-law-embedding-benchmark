#!/usr/bin/env python3
"""
Test append mode logic
"""

import os
import sys

# Mock các modules cần thiết để test logic
class MockQdrantClient:
    def __init__(self):
        self.collections = {}

    def recreate_collection(self, collection_name, vectors_config):
        self.collections[collection_name] = {
            'vectors_config': vectors_config,
            'points': []
        }
        print(f"Mock: Recreated collection {collection_name}")

def mock_get_collection_info(client, collection_name):
    """Mock version of get_collection_info"""
    if collection_name in client.collections:
        config = client.collections[collection_name]['vectors_config']
        return {
            'name': collection_name,
            'vectors_count': len(client.collections[collection_name]['points']),
            'points_count': len(client.collections[collection_name]['points']),
            'status': 'green',
            'config': {
                'params': {
                    'vectors': {
                        'size': config.size
                    }
                }
            }
        }
    return None

def mock_ensure_collection(client, collection_name, vector_size):
    """Mock version of ensure_collection"""
    client.recreate_collection(collection_name, type('MockConfig', (), {'size': vector_size})())
    print(f"Mock: Ensured collection {collection_name} with size {vector_size}")

def mock_ensure_or_append_collection(client, collection_name, vector_size, append_mode=False):
    """Mock version of ensure_or_append_collection"""
    existing_info = mock_get_collection_info(client, collection_name)

    if existing_info:
        if append_mode:
            # Kiểm tra vector size compatibility
            existing_vector_size = existing_info.get('config', {}).get('params', {}).get('vectors', {}).get('size', 0)
            if existing_vector_size != vector_size:
                raise ValueError(f"Vector size mismatch! Collection '{collection_name}' has size {existing_vector_size}, but new vectors have size {vector_size}")

            print(f"Append mode: Using existing collection '{collection_name}' ({existing_info['points_count']} existing points)")
            return False  # Đã tồn tại, append
        else:
            # Recreate mode
            print(f"Recreating existing collection '{collection_name}'...")
            mock_ensure_collection(client, collection_name, vector_size)
            return True  # Đã recreate
    else:
        # Collection chưa tồn tại
        print(f"Creating new collection '{collection_name}'...")
        mock_ensure_collection(client, collection_name, vector_size)
        return True  # Đã tạo mới

def test_append_mode():
    """Test append mode logic"""
    print("Testing append mode logic...")

    # Mock client
    client = MockQdrantClient()

    # Test 1: Collection chưa tồn tại, append mode = False (tạo mới)
    print("\n1. Test: Collection khong ton tai, append=False")
    try:
        result = mock_ensure_or_append_collection(client, "test_collection", 768, append_mode=False)
        print(f"   Result: {result} (True = created new)")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 2: Collection đã tồn tại, append mode = True (append)
    print("\n2. Test: Collection da ton tai, append=True")
    try:
        result = mock_ensure_or_append_collection(client, "test_collection", 768, append_mode=True)
        print(f"   Result: {result} (False = append mode)")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 3: Collection đã tồn tại, append mode = False (recreate)
    print("\n3. Test: Collection da ton tai, append=False")
    try:
        result = mock_ensure_or_append_collection(client, "test_collection", 768, append_mode=False)
        print(f"   Result: {result} (True = recreated)")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 4: Vector size mismatch
    print("\n4. Test: Vector size mismatch")
    try:
        result = mock_ensure_or_append_collection(client, "test_collection", 512, append_mode=True)
        print(f"   Result: {result}")
    except ValueError as e:
        print(f"   Expected error: {e}")

    print("\nAppend mode logic test completed!")

if __name__ == "__main__":
    test_append_mode()
