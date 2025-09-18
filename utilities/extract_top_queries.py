#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để extract top queries analysis từ ../results/embedding_evaluation_results.json
và tạo file JSON sạch chỉ chứa thông tin top queries
"""

import json

def extract_top_queries():
    """Extract top queries analysis từ results file"""

    print("🔄 Extracting top queries analysis...")

    try:
        # Load current results
        with open('../results/embedding_evaluation_results.json', 'r', encoding='utf-8') as f:
            results = json.load(f)

        print(f"✅ Loaded results for {len(results)} models")

        # Extract chỉ top queries analysis
        top_queries_summary = []

        for model_result in results:
            model_name = model_result['model_name']

            # Check if has top queries analysis
            if 'top_queries_by_total_score' in model_result:
                top_queries = model_result['top_queries_by_total_score']

                # Create clean summary for this model
                model_summary = {
                    "model_name": model_name,
                    "top_queries_by_total_score": top_queries
                }

                top_queries_summary.append(model_summary)
                print(f"✅ Extracted {len(top_queries)} top queries for {model_name}")
            else:
                print(f"⚠️ No top queries analysis found for {model_name}")

        # Save to new clean JSON file
        output_file = '../results/top_queries_analysis.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(top_queries_summary, f, ensure_ascii=False, indent=2)

        print(f"\\n🎉 Successfully created {output_file}")
        print(f"📊 Contains top queries analysis for {len(top_queries_summary)} models")

        # Show sample structure
        if top_queries_summary:
            sample_model = top_queries_summary[0]
            sample_queries = sample_model['top_queries_by_total_score']

            print(f"\\n📋 Sample structure for {sample_model['model_name']}:")
            if sample_queries:
                sample_query = sample_queries[0]
                print(f"   Query: {sample_query['query'][:60]}...")
                print(f"   Total Score: {sample_query['total_score']:.4f}")
                print(f"   Category: {sample_query['question_info'].get('category', 'N/A')}")

                top_answers = sample_query.get('top_answers', [])
                if top_answers:
                    top_answer = top_answers[0]
                    print(f"   Top Answer Score: {top_answer['score']:.4f}")
                    print(f"   Citation: {top_answer.get('citation', 'N/A')}")
                    print(f"   Content Preview: {top_answer.get('content', '')[:80]}...")

        print(f"\\n💡 File {output_file} ready for easy display and tracking!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    extract_top_queries()
