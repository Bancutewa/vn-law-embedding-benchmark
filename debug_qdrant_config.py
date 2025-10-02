#!/usr/bin/env python3
"""
Debug Qdrant collection config structure
"""

import os
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient

def debug_collection_config():
    """Debug cấu trúc collection config"""
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        print("ERROR: QDRANT_URL not set")
        return

    try:
        client = QdrantClient(
            url=url,
            api_key=api_key or None,
            timeout=30.0
        )

        # Lấy danh sách collections
        collections = client.get_collections()
        print(f"Available collections: {[col.name for col in collections.collections]}")

        # Nếu có collection nào đó, debug cấu trúc
        if collections.collections:
            collection_name = collections.collections[0].name
            print(f"\nDebugging collection: {collection_name}")

            try:
                info = client.get_collection(collection_name=collection_name)
                print("Collection info keys:", list(vars(info).keys()) if hasattr(info, '__dict__') else "No __dict__")

                if hasattr(info, 'config'):
                    print("Config type:", type(info.config))
                    print("Config attributes:", dir(info.config))

                    # Debug cấu trúc config
                    if hasattr(info.config, 'params'):
                        print("Has params:", hasattr(info.config, 'params'))
                        if hasattr(info.config.params, 'vectors'):
                            print("Has vectors:", hasattr(info.config.params, 'vectors'))
                            if hasattr(info.config.params.vectors, 'size'):
                                print("Vector size:", info.config.params.vectors.size)
                            else:
                                print("No size attribute in vectors")
                        else:
                            print("No vectors attribute in params")
                    else:
                        print("No params attribute in config")

                    # Thử dict access
                    try:
                        config_dict = vars(info.config)
                        print("Config as dict:", config_dict)
                    except:
                        print("Cannot convert config to dict")

            except Exception as e:
                print(f"Error getting collection info: {e}")

    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")

if __name__ == "__main__":
    debug_collection_config()</content>
</xai:function_call name="run_terminal_cmd">
<parameter name="command">python debug_qdrant_config.py
