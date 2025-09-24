#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding Models Manager
Quản lý các model embedding và encoding functions
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm

def mean_pooling(model_output, attention_mask):
    """Mean pooling để tạo sentence embeddings từ token embeddings"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_with_transformers(texts: List[str], model_name: str, max_length: int = 512, batch_size: int = 32, device: str = "cuda") -> np.ndarray:
    """Encode texts using transformers library với mean pooling"""
    from transformers import AutoTokenizer, AutoModel

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]

            # Tokenize
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)

            # Get model output
            model_output = model(**encoded)

            # Mean pooling
            sentence_embeddings = mean_pooling(model_output, encoded['attention_mask'])

            # Normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

            # Move to CPU and convert to numpy
            embeddings = sentence_embeddings.cpu().numpy()
            all_embeddings.append(embeddings)

            # Clear GPU memory
            del encoded, model_output, sentence_embeddings, embeddings
            torch.cuda.empty_cache() if device == "cuda" else None

    # Delete model to free memory
    del model, tokenizer
    torch.cuda.empty_cache() if device == "cuda" else None

    # Combine all embeddings
    final_embeddings = np.vstack(all_embeddings)
    print(f"   ✅ Generated {final_embeddings.shape[0]} embeddings")

    return final_embeddings

def encode_with_sentence_transformers(texts: List[str], model_name: str, batch_size: int = 32, device: str = "cuda") -> np.ndarray:
    """Encode texts using sentence-transformers library"""
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    model.to(device)

    # Encode all texts
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Delete model to free memory
    del model
    torch.cuda.empty_cache() if device == "cuda" else None

    print(f"   ✅ Generated {embeddings.shape[0]} embeddings")

    return embeddings

def encode_texts(texts: List[str], model_info: Dict[str, Any], device: str = "cuda") -> np.ndarray:
    """Encode texts sử dụng model được chỉ định"""
    model_type = model_info['type']
    model_name = model_info['name']

    if model_type == 'sentence_transformers':
        return encode_with_sentence_transformers(
            texts=texts,
            model_name=model_name,
            batch_size=16,
            device=device
        )
    else:  # transformers
        return encode_with_transformers(
            texts=texts,
            model_name=model_name,
            max_length=model_info.get('max_length', 512),
            batch_size=16,
            device=device
        )

def get_models_to_evaluate() -> List[Dict[str, Any]]:
    """Danh sách các mô hình cần đánh giá"""
    return [
        {
            'name': 'minhquan6203/paraphrase-vietnamese-law',
            'type': 'transformers',
            'description': 'Mô hình Sentence Similarity đã fine tune trên bộ luật pháp Việt Nam',
            'max_length': 512
        },
        {
            'name': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            'type': 'sentence_transformers',
            'description': 'Mô hình cơ sở đa ngôn ngữ (base model)',
            'max_length': 512
        },
        {
            'name': 'namnguyenba2003/Vietnamese_Law_Embedding_finetuned_v3_256dims',
            'type': 'transformers',
            'description': 'Mô hình embedding luật Việt Nam với 256 dimensions',
            'max_length': 512
        },
        {
            'name': 'truro7/vn-law-embedding',
            'type': 'sentence_transformers',
            'description': 'Mô hình embedding luật Việt Nam bằng Sentence Transformers',
            'max_length': 512
        }
    ]

def get_embedding_dimension(model_info: Dict[str, Any]) -> int:
    """Lấy dimension của embedding từ model (inference từ sample encoding)"""
    try:
        # Encode a sample text to get dimension
        sample_text = ["Điều kiện kết hôn theo pháp luật Việt Nam"]
        embeddings = encode_texts(sample_text, model_info, device="cpu")
        return embeddings.shape[1]
    except Exception as e:
        print(f"⚠️ Could not determine embedding dimension for {model_info['name']}: {e}")
        # Default dimensions based on model type
        if '256dims' in model_info['name']:
            return 256
        elif 'minilm' in model_info['name'].lower():
            return 384
        else:
            return 768  # Most common default
