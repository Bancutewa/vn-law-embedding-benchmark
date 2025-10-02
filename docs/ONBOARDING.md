# Vietnamese Law Embedding Benchmark – Onboarding Guide

## 1. Project mission
This repository compares multiple embedding models on Vietnamese legal corpora, aiming to identify which model retrieves the most relevant statute passages for benchmark questions. It also includes tools to prepare raw DOC/DOCX laws, sample benchmark questions, and visualize top-scoring retrievals via Qdrant. 【F:README.md†L1-L43】【F:README_refactored.md†L1-L88】

## 2. Repository layout
The project ships with both the original monolithic workflow and a refactored, modular pipeline. Key top-level assets include:

- **`embedding_evaluation.py`** – legacy end-to-end script that still works but packs ~1,900 lines of logic into a single file. 【F:README_refactored.md†L101-L136】
- **`main_refactored.py`** – entry point for the modular pipeline; it sets up the environment, loads prepared chunks and benchmark queries, runs evaluation across the registered models, and prints a detailed report. 【F:main_refactored.py†L1-L191】
- **`modules/`** – refactored building blocks that encapsulate chunking, loading, embedding, Qdrant operations, and evaluation routines. 【F:README_refactored.md†L89-L136】
- **`data_files/`** – cached metadata such as discovered law file paths and extracted benchmark questions. These JSON files are produced by discovery scripts and reused during chunking/evaluation. 【F:README.md†L45-L63】
- **`data/`** – generated chunk files (per category or full corpus) ready for embedding and evaluation. 【F:README_refactored.md†L89-L176】
- **`results/`** – evaluation artifacts such as aggregated metrics and top-query analyses. 【F:README.md†L45-L63】

Support scripts like `find_law_files.py`, `find_question_files.py`, `chunking.py`, and `embed_and_upload.py` orchestrate the data preparation pipeline and model upload steps. 【F:README.md†L45-L63】【F:README_refactored.md†L1-L176】

## 3. Data preparation workflow
1. **Catalog source statutes:** Run `find_law_files.py` to build `data_files/law_file_paths.json` and, in the refactored flow, category-specific JSON manifests (BDS, DN, TM, QDS). 【F:README.md†L45-L63】【F:README_refactored.md†L9-L40】
2. **Extract benchmark questions:** Run `find_question_files.py` to transform Excel sheets into `data_files/law_questions.json`, which holds 511 labeled questions. 【F:README.md†L7-L43】【F:README_refactored.md†L41-L68】
3. **Chunk the statutes:** Use `chunking.py` to convert DOC/DOCX laws into structured chunks. The module handles Unicode normalization, clause-aware heuristics, law ID generation, and optional Gemini-based quality review. Outputs land in `data/` with filenames scoped by category and timestamp. 【F:modules/chunking.py†L1-L120】【F:modules/chunking.py†L121-L240】【F:README_refactored.md†L41-L176】
4. **(Optional) Upload to Qdrant:** `embed_and_upload.py` encodes a chosen chunk file with a specific model and stores vectors in Qdrant collections named by model/category. 【F:README_refactored.md†L129-L176】【F:modules/embedding_models.py†L1-L120】【F:modules/qdrant_manager.py†L1-L120】

## 4. Evaluation workflow
- **Load data:** `modules/data_loader` reads the law manifests, reuses chunk metadata, and samples benchmark questions (with automatic fallbacks if JSON is missing). It enriches chunk metadata with source file provenance for downstream inspection. 【F:modules/data_loader.py†L1-L200】
- **Encode & store vectors:** `modules/embedding_models` wraps both `transformers` and `sentence-transformers` models with batching, mean pooling, and normalization, while `modules/qdrant_manager` recreates collections, manages payload indexes, and upserts vectors with retry logic. 【F:modules/embedding_models.py†L1-L160】【F:modules/qdrant_manager.py†L1-L160】
- **Score retrieval quality:** `modules/evaluation` connects to Qdrant, encodes queries, performs vector search, collects per-query metrics (max, top-k averages, threshold counts), and aggregates model-level summaries. It also prints rich console reports and optionally reuses existing Qdrant collections to skip re-encoding. 【F:modules/evaluation.py†L1-L200】
- **Run orchestrator:** `main_refactored.py` calls the above modules end-to-end, then formats a ranking table, highlights best-performing models, and surfaces top queries with supporting metadata excerpts. 【F:main_refactored.py†L1-L191】

## 5. Key concepts & conventions
- **Law IDs:** Automatically generated identifiers distinguish primary statutes from amendments (e.g., `LKBDS`, `LSĐBSLKBDS`) to keep metadata consistent across chunks, embeddings, and Qdrant indexes. 【F:README_refactored.md†L69-L110】【F:modules/chunking.py†L121-L240】
- **Chunk metadata:** Every chunk carries citation fields (law ID, article, clause) plus source provenance so analysts can trace retrieval hits back to original documents. Qdrant payload indexes are created for these metadata fields to enable filtering. 【F:modules/chunking.py†L160-L240】【F:modules/qdrant_manager.py†L33-L120】
- **Model registry:** `get_models_to_evaluate()` centralizes the list of supported embedding checkpoints, ensuring both pipelines evaluate the same baseline models unless you modify the registry. 【F:modules/embedding_models.py†L121-L180】

## 6. Recommended next steps for newcomers
1. **Run the refactored pipeline:** Follow the workflow in `README_refactored.md` (setup → discovery → chunking → `main_refactored.py`) to observe the end-to-end evaluation output locally. 【F:README_refactored.md†L101-L176】
2. **Inspect sample results:** Explore `results/embedding_evaluation_results.json` and `results/top_queries_analysis.json` to understand how metrics and retrieved passages are stored. 【F:README.md†L45-L63】
3. **Experiment with models:** Add a new entry to `get_models_to_evaluate()` and rerun the evaluation to see how alternative embeddings behave. 【F:modules/embedding_models.py†L121-L180】
4. **Deep-dive into chunking heuristics:** Review `modules/chunking.py` to learn how Vietnamese statute structure is parsed; tweaking heuristics can improve retrieval quality. 【F:modules/chunking.py†L1-L200】
5. **Integrate with Qdrant filters:** Use the payload indexes created in `qdrant_manager.py` to build filtered search or analytics tools that target specific laws, chapters, or categories. 【F:modules/qdrant_manager.py†L33-L160】

Happy benchmarking! 🚀
