#!/usr/bin/env python3
"""
Test collection logic riêng biệt
"""

import sys
import os

# Mock các dependencies
class MockCollectionConfig:
    def __init__(self):
        self.params = MockParams()

class MockParams:
    def __init__(self):
        self.vectors = MockVectors()

class MockVectors:
    def __init__(self):
        self.size = 768

# Import function cần test
sys.path.append(os.path.dirname(__file__))
from modules.qdrant_manager import ensure_or_append_collection

# Mock client
class MockClient:
    def __init__(self):
        self.collections = {}

    def recreate_collection(self, name, config):
        self.collections[name] = {
            'config': MockCollectionConfig(),
            'vectors_count': 0,
            'points_count': 0
        }

    def get_collection(self, name):
        if name in self.collections:
            return self.collections[name]
        return None

def mock_get_collection_info(client, collection_name):
    collection = client.get_collection(collection_name)
    if collection:
        return {
            'name': collection_name,
            'config': collection['config'],
            'vectors_count': collection['vectors_count'],
            'points_count': collection['points_count']
        }
    return None

# Override get_collection_info trong qdrant_manager
import modules.qdrant_manager
modules.qdrant_manager.get_collection_info = mock_get_collection_info

def test_collection_logic():
    print("Testing collection logic...")

    client = MockClient()

    # Test 1: Collection không tồn tại, append mode
    print("\n1. Test: Collection khong ton tai, append=True")
    try:
        result = ensure_or_append_collection(client, "test_collection", 768, append_mode=True)
        print(f"   Result: {result} (True = created new)")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 2: Collection đã tồn tại, append mode
    print("\n2. Test: Collection da ton tai, append=True")
    try:
        result = ensure_or_append_collection(client, "test_collection", 768, append_mode=True)
        print(f"   Result: {result} (False = append mode)")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 3: Collection đã tồn tại, vector size mismatch
    print("\n3. Test: Vector size mismatch")
    try:
        result = ensure_or_append_collection(client, "test_collection", 512, append_mode=True)
        print(f"   Result: {result}")
    except ValueError as e:
        print(f"   Expected error: {e}")

    print("\nCollection logic test completed!")

if __name__ == "__main__":
    test_collection_logic()
