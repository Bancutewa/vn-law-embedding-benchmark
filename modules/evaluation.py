#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Module
Logic đánh giá chất lượng các embedding models
"""

import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .qdrant_manager import get_qdrant_client, ensure_collection, upsert_embeddings_to_qdrant, search_qdrant, count_collection_points
from .embedding_models import encode_texts, get_embedding_dimension
from .data_loader import display_search_results

def calculate_metrics(scores: np.ndarray, threshold_07: float = 0.7, threshold_05: float = 0.5) -> Dict[str, float]:
    """Tính toán các metrics đánh giá chất lượng retrieval"""
    return {
        "max_score": float(np.max(scores)),
        "avg_top3": float(np.mean(scores[:3])) if len(scores) >= 3 else float(np.mean(scores)),
        "avg_top5": float(np.mean(scores[:5])) if len(scores) >= 5 else float(np.mean(scores)),
        "avg_top10": float(np.mean(scores[:10])) if len(scores) >= 10 else float(np.mean(scores)),
        "avg_all": float(np.mean(scores)),
        "scores_above_07": int(np.sum(scores >= threshold_07)),
        "scores_above_05": int(np.sum(scores >= threshold_05)),
        "min_score": float(np.min(scores))
    }

def evaluate_single_model(model_info: Dict[str, Any], law_docs: List[Dict[str, Any]], queries: List[str], top_k: int = 15, show_detailed_results: bool = True, device: str = "cuda") -> Dict[str, Any]:
    """Đánh giá một mô hình embedding"""
    print(f"\n{'='*80}")
    print(f"🔍 EVALUATING MODEL: {model_info['name']}")
    print(f"📝 Description: {model_info['description']}")
    print(f"🔧 Type: {model_info['type']} | Max Length: {model_info['max_length']} tokens")
    print(f"{'='*80}")

    try:
        # Bước 1: Chuẩn bị texts
        doc_texts = [doc['content'] for doc in law_docs]
        print(f"\n📚 Step 1: Prepared {len(doc_texts)} document texts")

        # Bước 2: Kiểm tra Qdrant collection trước
        client = get_qdrant_client()
        collection_name = model_info['name'].replace('/', '_')
        existing = count_collection_points(client, collection_name)

        if existing >= len(law_docs):
            print(f"🟡 Collection '{collection_name}' already has {existing} vectors (>= {len(law_docs)}). Skipping document encoding.")
        else:
            print(f"🟠 Collection '{collection_name}' has {existing} vectors. Need to encode and upsert {len(law_docs)} vectors...")

            # Bước 2a: Encode documents
            print(f"\n🔨 Step 2a: Encoding documents...")
            doc_embeddings = encode_texts(doc_texts, model_info, device=device)
            print(f"   ✅ Document embeddings shape: {doc_embeddings.shape}")

            # Bước 2b: Lưu vào Qdrant
            print(f"\n💾 Step 2b: Storing embeddings in Qdrant...")
            vector_size = doc_embeddings.shape[1]
            ensure_collection(client, collection_name, vector_size)
            upsert_embeddings_to_qdrant(client, collection_name, doc_embeddings, law_docs)

            # Giải phóng RAM
            del doc_embeddings

        # Bước 3: Encode queries
        print(f"\n🔍 Step 3: Encoding queries...")
        query_embeddings = encode_texts(queries, model_info, device=device)
        print(f"   ✅ Query embeddings shape: {query_embeddings.shape}")

        # Bước 4: Evaluate từng query
        print(f"\n📊 Step 4: Evaluating {len(queries)} queries...")
        query_results = []
        all_metrics = []

        for i, query in enumerate(queries):
            print(f"\n   🔍 Query {i+1}/{len(queries)}")

            # Search top-k documents via Qdrant
            top_indices, top_scores = search_qdrant(
                client=client,
                collection_name=collection_name,
                query_embedding=query_embeddings[i],
                top_k=top_k
            )

            # Calculate metrics
            metrics = calculate_metrics(top_scores)
            all_metrics.append(metrics)

            # Store results
            query_result = {
                'query': query,
                'query_id': i,
                'top_indices': top_indices.tolist(),
                'top_scores': top_scores.tolist(),
                'metrics': metrics
            }
            query_results.append(query_result)

            # Show detailed results for first few queries
            if show_detailed_results and i < 3:
                display_search_results(query, law_docs, top_indices, top_scores, max_display=3)
                print(f"      📈 Metrics: Max={metrics['max_score']:.4f}, Avg_top5={metrics['avg_top5']:.4f}, Above_0.7={metrics['scores_above_07']}")

        # Bước 5: Aggregate metrics
        print(f"\n📈 Step 5: Aggregating metrics...")

        # Calculate average metrics across all queries
        avg_metrics = {}
        metric_keys = all_metrics[0].keys()
        for key in metric_keys:
            if key.startswith('scores_above'):
                avg_metrics[f"avg_{key}"] = np.mean([m[key] for m in all_metrics])
            else:
                avg_metrics[f"avg_{key}"] = np.mean([m[key] for m in all_metrics])

        # Final result
        final_result = {
            'model_name': model_info['name'],
            'model_type': model_info['type'],
            'model_description': model_info['description'],
            'max_length': model_info['max_length'],
            'num_queries': len(queries),
            'num_documents': len(law_docs),
            'top_k': top_k,
            'query_results': query_results,
            'aggregated_metrics': avg_metrics,
            'evaluation_success': True
        }

        # Print summary
        print(f"\n✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print(f"   📊 Average Results:")
        print(f"      - Avg Max Score: {avg_metrics['avg_max_score']:.4f}")
        print(f"      - Avg Top-5 Score: {avg_metrics['avg_avg_top5']:.4f}")
        print(f"      - Avg Above 0.7: {avg_metrics['avg_scores_above_07']:.1f}")
        print(f"      - Avg Above 0.5: {avg_metrics['avg_scores_above_05']:.1f}")

        return final_result

    except Exception as e:
        print(f"\n❌ EVALUATION FAILED: {str(e)}")
        return {
            'model_name': model_info['name'],
            'model_type': model_info['type'],
            'error': str(e),
            'evaluation_success': False
        }

def run_evaluation_all_models(models_to_evaluate: List[Dict[str, Any]], law_docs: List[Dict[str, Any]], benchmark_queries: List[str], device: str = "cuda") -> List[Dict[str, Any]]:
    """Chạy đánh giá cho tất cả các mô hình"""
    print("🚀 Starting evaluation for all models...")

    if not law_docs:
        print("❌ Error: No law documents loaded!")
        return []

    print(f"✅ Ready to evaluate with:")
    print(f"   📚 Documents: {len(law_docs)} law chunks")
    print(f"   ❓ Queries: {len(benchmark_queries)} benchmark questions")
    print(f"   🤖 Models: {len(models_to_evaluate)} models to test")
    print(f"   🎯 Top-K: 15 results per query")

    evaluation_results = []
    successful_evaluations = 0
    failed_evaluations = 0

    for i, model_info in enumerate(models_to_evaluate):
        print(f"\n{'🤖 '*20}")
        print(f"🤖 EVALUATING MODEL {i+1}/{len(models_to_evaluate)}")
        print(f"{'🤖 '*20}")

        try:
            result = evaluate_single_model(
                model_info=model_info,
                law_docs=law_docs,
                queries=benchmark_queries,
                top_k=15,
                show_detailed_results=(i < 2),  # Show details for first 2 models only
                device=device
            )

            if result['evaluation_success']:
                evaluation_results.append(result)
                successful_evaluations += 1
                print(f"✅ Model {i+1} evaluation completed successfully!")
            else:
                print(f"❌ Model {i+1} evaluation failed: {result.get('error', 'Unknown error')}")
                failed_evaluations += 1

            # Wait between models
            if i < len(models_to_evaluate) - 1:
                print(f"⏳ Waiting 2 seconds before next model...")
                import time
                time.sleep(2)

        except Exception as e:
            print(f"❌ Unexpected error evaluating model {i+1}: {str(e)}")
            failed_evaluations += 1
            continue

    # Final summary
    print(f"\n{'='*100}")
    print(f"🎉 EVALUATION SUMMARY")
    print(f"{'='*100}")
    print(f"✅ Successful evaluations: {successful_evaluations}")
    print(f"❌ Failed evaluations: {failed_evaluations}")
    print(f"📊 Total models evaluated: {len(evaluation_results)}")

    if evaluation_results:
        print(f"\n📈 Quick Performance Preview:")
        # Sắp xếp theo avg_max_score giảm dần
        preview_sorted = sorted(
            evaluation_results,
            key=lambda r: r['aggregated_metrics']['avg_max_score'],
            reverse=True
        )
        for result in preview_sorted:
            metrics = result['aggregated_metrics']
            model_name = result['model_name'].split('/')[-1]
            print(f"   🤖 {model_name}")
            print(f"      Max Score: {metrics['avg_max_score']:.4f} | Top-5: {metrics['avg_avg_top5']:.4f} | Above 0.7: {metrics['avg_scores_above_07']:.1f}")

    return evaluation_results
