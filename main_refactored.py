#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vietnamese Law Embedding Benchmark - Refactored Version
Đánh giá các mô hình embedding cho luật tiếng Việt
"""

import os
import sys
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import các modules đã tách
from modules.data_loader import load_question_benchmark, load_chunks_from_json
from modules.embedding_models import get_models_to_evaluate
from modules.evaluation import run_evaluation_all_models, evaluate_single_model
from modules.qdrant_manager import get_qdrant_client

def setup_environment():
    """Thiết lập môi trường và import các thư viện cần thiết"""
    print("🔄 Setting up environment...")

    # Check GPU availability
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\n🎉 Environment ready!")
    return device

def generate_final_report(evaluation_results, law_docs, benchmark_queries, sample_questions=None):
    """Tạo báo cáo chi tiết cuối cùng"""
    print("📊 Generating detailed analysis and final report...")

    if not evaluation_results:
        print("❌ No evaluation results available!")
        return None

    print(f"\n{'='*100}")
    print(f"📋 COMPREHENSIVE EVALUATION REPORT")
    print(f"{'='*100}")

    # Sort models by average max score (best first)
    sorted_results = sorted(
        evaluation_results,
        key=lambda x: x['aggregated_metrics']['avg_max_score'],
        reverse=True
    )

    print(f"📊 DATASET INFORMATION:")
    print(f"   📚 Law Documents: {len(law_docs)} chunks from Vietnamese Law")
    print(f"   ❓ Benchmark Queries: {len(benchmark_queries)} questions")
    print(f"   🔍 Evaluation Method: Top-15 retrieval with cosine similarity")
    print(f"   💾 Storage: Qdrant vector database")

    print(f"\n🏆 RANKING BY PERFORMANCE:")
    print(f"   Metric: Average Max Score across all queries")

    # Create comparison table
    print(f"\n{'Rank':<4} {'Model':<45} {'Max Score':<10} {'Top-5':<8} {'≥0.7':<6} {'≥0.5':<6} {'Type':<12}")
    print(f"{'-'*95}")

    for i, result in enumerate(sorted_results):
        metrics = result['aggregated_metrics']
        model_name = result['model_name'].split('/')[-1][:40]
        model_type = result['model_type']

        print(f"{i+1:<4} {model_name:<45} {metrics['avg_max_score']:<10.4f} "
              f"{metrics['avg_avg_top5']:<8.4f} {metrics['avg_scores_above_07']:<6.1f} "
              f"{metrics['avg_scores_above_05']:<6.1f} {model_type:<12}")

    # Best model analysis
    best_model = sorted_results[0]
    print(f"\n⭐ RECOMMENDED MODEL:")
    print(f"   🥇 {best_model['model_name']}")
    print(f"   📝 {best_model['model_description']}")
    print(f"   🎯 Performance Highlights:")
    best_metrics = best_model['aggregated_metrics']
    print(f"      - Average Max Score: {best_metrics['avg_max_score']:.4f}")
    print(f"      - Average Top-5 Score: {best_metrics['avg_avg_top5']:.4f}")
    print(f"      - Queries with score ≥ 0.7: {best_metrics['avg_scores_above_07']:.1f} per query")
    print(f"      - Queries with score ≥ 0.5: {best_metrics['avg_scores_above_05']:.1f} per query")

    # Performance analysis
    print(f"\n📈 PERFORMANCE ANALYSIS:")

    # Calculate overall statistics
    all_max_scores = [r['aggregated_metrics']['avg_max_score'] for r in evaluation_results]
    all_top5_scores = [r['aggregated_metrics']['avg_avg_top5'] for r in evaluation_results]

    print(f"   📊 Overall Statistics:")
    print(f"      - Best Max Score: {max(all_max_scores):.4f}")
    print(f"      - Worst Max Score: {min(all_max_scores):.4f}")
    print(f"      - Average Max Score: {np.mean(all_max_scores):.4f}")
    print(f"      - Best Top-5 Score: {max(all_top5_scores):.4f}")
    print(f"      - Average Top-5 Score: {np.mean(all_top5_scores):.4f}")

    # Model type analysis
    transformers_models = [r for r in evaluation_results if r['model_type'] == 'transformers']
    sentence_transformers_models = [r for r in evaluation_results if r['model_type'] == 'sentence_transformers']

    if transformers_models and sentence_transformers_models:
        print(f"\n🔧 MODEL TYPE COMPARISON:")

        trans_avg = np.mean([r['aggregated_metrics']['avg_max_score'] for r in transformers_models])
        sent_avg = np.mean([r['aggregated_metrics']['avg_max_score'] for r in sentence_transformers_models])

        print(f"   🔨 Transformers models: {len(transformers_models)} models, avg score: {trans_avg:.4f}")
        print(f"   📦 Sentence-transformers: {len(sentence_transformers_models)} models, avg score: {sent_avg:.4f}")

        if trans_avg > sent_avg:
            print(f"   ✅ Transformers models perform better on average (+{trans_avg - sent_avg:.4f})")
        else:
            print(f"   ✅ Sentence-transformers models perform better on average (+{sent_avg - trans_avg:.4f})")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   🎯 For Production Deployment:")
    print(f"      - Primary: {best_model['model_name']}")
    print(f"      - Type: {best_model['model_type']}")
    print(f"      - Max Length: {best_model['max_length']} tokens")

    if len(sorted_results) > 1:
        second_best = sorted_results[1]
        print(f"   🥈 Alternative Option:")
        print(f"      - {second_best['model_name']}")
        print(f"      - Performance difference: {best_metrics['avg_max_score'] - second_best['aggregated_metrics']['avg_max_score']:.4f}")

    print(f"\n🔍 DETAILED QUERY ANALYSIS:")
    print(f"   📝 Sample Query Performance (Best Model):")

    # Show performance on first 3 queries for best model
    best_query_results = best_model['query_results'][:3]
    for i, qr in enumerate(best_query_results):
        print(f"\n   Query {i+1}: {qr['query'][:60]}...")
        print(f"      Max Score: {qr['metrics']['max_score']:.4f}")
        print(f"      Top-3 Average: {qr['metrics']['avg_top3']:.4f}")
        print(f"      Results ≥ 0.7: {qr['metrics']['scores_above_07']}")

    # Top queries by total score (best model)
    print(f"\n🔝 TOP 3 QUERIES BY TOTAL SCORE (Best Model)")
    # Compute total score = sum of retrieved top_k scores for each query
    query_totals = []
    for qr in best_model['query_results']:
        total = float(np.sum(qr.get('top_scores', [])))
        query_totals.append({
            'query_id': qr['query_id'],
            'query': qr['query'],
            'total_score': total,
            'top_indices': qr.get('top_indices', []),
            'top_scores': qr.get('top_scores', [])
        })
    # Sort and take top 3
    query_totals.sort(key=lambda x: x['total_score'], reverse=True)
    top3_queries = query_totals[:3]

    # Build detailed objects and print
    top_queries_detailed = []
    for rank, q in enumerate(top3_queries, start=1):
        print(f"\n   {rank}. Total Score: {q['total_score']:.4f}")
        print(f"      Query: {q['query']}")
        # Hiển thị thông tin question từ Excel nếu có
        question_info = {}
        if sample_questions:
            # Find matching question by query text to ensure correct mapping
            query_text = q['query'].strip()
            for sq in sample_questions:
                if sq['primary_query'].strip() == query_text:
                    question_info = {
                        'category': sq.get('full_category', ''),
                        'positive_answer': sq.get('positive', ''),
                        'negative_answer': sq.get('negative', ''),
                        'source_file': sq.get('source_file', '')
                    }
                    print(f"      Category: {question_info['category']}")
                    if question_info['positive_answer']:
                        print(f"      Expected Answer: {question_info['positive_answer'][:100]}...")
                    if question_info['negative_answer']:
                        print(f"      Negative Answer: {question_info['negative_answer'][:100]}...")
                    break

        # Print top 3 results for this query with FULL CONTENT
        top_n = min(3, len(q['top_indices']))
        answers = []
        for i in range(top_n):
            idx = int(q['top_indices'][i])
            score = float(q['top_scores'][i])
            if 0 <= idx < len(law_docs):
                doc = law_docs[idx]
                md = doc.get('metadata') or {}
                citation = md.get('exact_citation') or ''
                law_id = md.get('law_id', '')

                print(f"\n         📄 Rank {i+1}: Score {score:.4f} | Law: {law_id}")
                print(f"         📝 Citation: {citation}")
                print(f"         📖 Content: {doc['content'][:300]}...")
                if len(doc['content']) > 300:
                    print(f"         ... ({len(doc['content'])} chars total)")

                # collect for JSON
                answers.append({
                    'rank': i+1,
                    'score': score,
                    'citation': citation,
                    'law_id': law_id,
                    'content': doc.get('content', ''),
                    'metadata': md
                })
            else:
                print(f"         - Rank {i+1}: {score:.4f} | [Index {idx} out of range]")
                answers.append({
                    'rank': i+1,
                    'score': score,
                    'citation': '',
                    'law_id': '',
                    'content': '',
                    'metadata': {},
                    'note': f'Index {idx} out of range'
                })
        # Thêm thông tin question từ Excel (đã được set ở trên)

        top_queries_detailed.append({
            'rank': rank,
            'query_id': q['query_id'],
            'query': q['query'],
            'total_score': q['total_score'],
            'question_info': question_info,
            'top_answers': answers
        })

    # Attach into the best model object for JSON output
    try:
        best_model['top_queries_by_total_score'] = top_queries_detailed
    except Exception:
        pass

    # Save summary
    evaluation_summary = {
        'best_model': best_model['model_name'],
        'best_score': best_metrics['avg_max_score'],
        'total_models_evaluated': len(evaluation_results),
        'total_queries': len(benchmark_queries),
        'total_documents': len(law_docs),
        'sorted_results': sorted_results,
        # attach preview for convenience
        'top_queries_preview': [
            {
                'query_id': q['query_id'],
                'query': q['query'],
                'total_score': q['total_score'],
                'top3': [
                    {
                        'doc_index': int(q['top_indices'][i]),
                        'score': float(q['top_scores'][i])
                    } for i in range(min(3, len(q['top_indices'])))
                ]
            } for q in top3_queries
        ],
        'top_queries_by_total_score': top_queries_detailed
    }

    print(f"\n{'='*100}")
    print(f"✅ EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"📊 Summary saved to evaluation_summary")
    print(f"📋 Full results available in evaluation_results")
    print(f"{'='*100}")

    # Export option
    print(f"\n💾 EXPORT OPTIONS:")
    print(f"   To save results to JSON file:")
    print(f"   → import json")
    print(f"   → with open('results/embedding_evaluation_results.json', 'w', encoding='utf-8') as f:")
    print(f"   →     json.dump(evaluation_results, f, ensure_ascii=False, indent=2)")

    print(f"\n🎉 Report generation completed!")

    return evaluation_summary

def test_single_query(best_model_name, models_to_evaluate, law_docs, device="cuda"):
    """Test với một query đơn lẻ"""
    print("🧪 Quick test with single query...")

    test_query = "Dự án đầu tư xây dựng khu đô thị phải có công năng hỗn hợp, đồng bộ hạ tầng hạ tầng kỹ thuật, hạ tầng xã hội và nhà ở theo quy hoạch được phê duyệt?"
    print(f"📝 Test Query: {test_query}")
    print(f"🥇 Using best model: {best_model_name}")

    # Find model info
    best_model_info = None
    for model in models_to_evaluate:
        if model['name'] == best_model_name:
            best_model_info = model
            break

    if best_model_info:
        print(f"🔍 Testing single query with model `{best_model_info['name']}`...")

        # Encode query
        from modules.embedding_models import encode_texts
        test_query_embedding = encode_texts([test_query], best_model_info, device=device)[0]

        # Search trực tiếp trên Qdrant (documents đã được upsert khi evaluate)
        client = get_qdrant_client()
        collection_name = best_model_info['name'].replace('/', '_')
        from modules.qdrant_manager import search_qdrant
        top_indices, top_scores = search_qdrant(client, collection_name, test_query_embedding, top_k=10)

        # Display results
        from modules.data_loader import display_search_results
        display_search_results(test_query, law_docs, top_indices, top_scores, max_display=5)

        # Metrics
        from modules.evaluation import calculate_metrics
        metrics = calculate_metrics(top_scores)
        print(f"\n📊 Metrics for this query:")
        print(f"   - Max Score: {metrics['max_score']:.4f}")
        print(f"   - Top-5 Average: {metrics['avg_top5']:.4f}")
        print(f"   - Results ≥ 0.7: {metrics['scores_above_07']}")
        print(f"   - Results ≥ 0.5: {metrics['scores_above_05']}")
    else:
        print("❌ Could not find model info for testing")

    print(f"\n✅ Single query test completed!")

def main():
    """Hàm chính chạy toàn bộ evaluation"""
    print("=" * 80)
    print("🔬 ĐÁNH GIÁ MÔ HÌNH EMBEDDING CHO LUẬT TIẾNG VIỆT (REFACTORED)")
    print("=" * 80)

    # 1. Setup environment
    device = setup_environment()

    # 2. Load chunks từ JSON file (đã được tạo bởi chunking.py)
    chunks_json_path = "data/chunks.json"
    print(f"\n📖 Loading chunks from {chunks_json_path}...")

    if not os.path.exists(chunks_json_path):
        print(f"❌ Chunks file not found: {chunks_json_path}")
        print("   Please run chunking.py first to create the chunks file")
        print("   Command: python chunking.py")
        return

    law_docs = load_chunks_from_json(chunks_json_path)
    if not law_docs:
        print("❌ Cannot proceed without law chunks!")
        return

    print(f"✅ Loaded {len(law_docs)} chunks from {chunks_json_path}")

    # 3. Get models and queries
    models_to_evaluate = get_models_to_evaluate()
    benchmark_queries, sample_questions = load_question_benchmark()  # Get benchmark queries and sample questions

    print(f"\n🤖 Prepared {len(models_to_evaluate)} models for evaluation:")
    for i, model in enumerate(models_to_evaluate):
        print(f"   {i+1}. {model['name']}")
        print(f"      Type: {model['type']} | Max Length: {model['max_length']} tokens")
        print(f"      Description: {model['description']}")
        print()

    print(f"🎯 All models support ≥512 tokens as required!")
    print(f"💾 Chunks loaded from JSON, embeddings will be stored in Qdrant vector database")

    print(f"\nPrepared {len(benchmark_queries)} benchmark queries from Excel files")
    print("Sample queries:")
    for i, query in enumerate(benchmark_queries[:5]):
        print(f"{i+1}. {query}")

    # 4. Run evaluation for all models (encodes chunks and uploads to Qdrant)
    evaluation_results = run_evaluation_all_models(models_to_evaluate, law_docs, benchmark_queries, device)

    # 5. Generate final report
    evaluation_summary = generate_final_report(evaluation_results, law_docs, benchmark_queries, sample_questions)

    # 6. Test single query with best model
    if evaluation_summary:
        test_single_query(evaluation_summary['best_model'], models_to_evaluate, law_docs, device)

    # 7. Save results to JSON
    if evaluation_results:
        try:
            # Save full results
            with open('results/embedding_evaluation_results.json', 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Full results saved to: results/embedding_evaluation_results.json")

            # Also save clean top queries analysis
            top_queries_summary = []
            for model_result in evaluation_results:
                model_name = model_result['model_name']
                if 'top_queries_by_total_score' in model_result:
                    top_queries_summary.append({
                        "model_name": model_name,
                        "top_queries_by_total_score": model_result['top_queries_by_total_score']
                    })

            if top_queries_summary:
                with open('results/top_queries_analysis.json', 'w', encoding='utf-8') as f:
                    json.dump(top_queries_summary, f, ensure_ascii=False, indent=2)
                print(f"💾 Top queries analysis saved to: results/top_queries_analysis.json")

        except Exception as e:
            print(f"\n⚠️ Could not save results to JSON: {e}")

    print(f"\n🎉 EVALUATION COMPLETED! 🎉")

if __name__ == "__main__":
    main()
