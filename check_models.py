#!/usr/bin/env python3
"""
Check available Gemini models
"""

import google.generativeai as genai
import os

# Try to get API key from env
api_key = os.getenv('GEMINI_API_KEY', '')

if not api_key:
    print("ERROR: No GEMINI_API_KEY found in environment")
    print("   Please set GEMINI_API_KEY in .env file or environment variable")
    exit(1)

try:
    genai.configure(api_key=api_key)
    models = genai.list_models()

    print("Available Gemini models:")
    gemini_models = [m for m in models if 'gemini' in m.name.lower()]
    for model in gemini_models:
        methods = model.supported_generation_methods
        print(f"   - {model.name}: {methods}")

    if not gemini_models:
        print("   WARNING: No Gemini models found")

except Exception as e:
    print(f"ERROR checking models: {e}")
    print("   Make sure GEMINI_API_KEY is valid")
