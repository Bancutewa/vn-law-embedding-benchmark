#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để xem top queries analysis trong ../results/embedding_evaluation_results.json
"""

import json

def view_top_queries():
    """Hiển thị top queries analysis từ file results"""

    print("📊 TOP QUERIES ANALYSIS VIEWER")
    print("=" * 50)

    try:
        # Load results
        with open('../results/embedding_evaluation_results.json', 'r', encoding='utf-8') as f:
            results = json.load(f)

        print(f"✅ Loaded results for {len(results)} models\\n")

        # Show analysis for each model
        for i, model in enumerate(results, 1):
            model_name = model['model_name']
            print(f"🏆 MODEL {i}: {model_name}")
            print("-" * 50)

            if 'top_queries_by_total_score' not in model:
                print("❌ No top queries analysis found")
                continue

            top_queries = model['top_queries_by_total_score']
            print(f"✅ Found {len(top_queries)} top queries\\n")

            # Show each top query
            for query in top_queries:
                rank = query['rank']
                total_score = query['total_score']
                query_text = query['query']

                print(f"   {rank}. Total Score: {total_score:.4f}")
                print(f"      Query: {query_text}")

                # Show question info
                question_info = query.get('question_info', {})
                if question_info.get('category'):
                    print(f"      Category: {question_info['category']}")

                if question_info.get('positive_answer'):
                    positive = question_info['positive_answer']
                    print(f"      Expected: {positive[:80]}{'...' if len(positive) > 80 else ''}")

                # Show top answer
                top_answers = query.get('top_answers', [])
                if top_answers:
                    top_answer = top_answers[0]
                    content = top_answer.get('content', '')
                    citation = top_answer.get('citation', '')
                    law_id = top_answer.get('law_id', '')

                    print(f"      Top Answer (Score: {top_answer['score']:.4f}):")
                    print(f"         Law: {law_id} | Citation: {citation}")
                    print(f"         Content: {content[:150]}{'...' if len(content) > 150 else ''}")

                print()

        print("=" * 50)
        print("✅ Top queries analysis displayed successfully!")
        print("\\n💡 File ../results/embedding_evaluation_results.json now contains:")
        print("   - 50 random queries evaluation")
        print("   - Top 3 queries by total score for each model")
        print("   - Full content with citation and metadata")
        print("   - Question context from Excel files")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    view_top_queries()
