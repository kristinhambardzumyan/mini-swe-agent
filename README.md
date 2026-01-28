# Retrieval-Augmented Code Generation (RACG)

**University Final Project**

This repository is a fork of [`SWE-agent/mini-swe-agent`](https://github.com/SWE-agent/mini-swe-agent),
extended for a university final project on **Retrieval-Augmented Code Generation (RACG)**.

We study whether retrieving relevant code snippets from a repository improves
LLM-based program repair under execution-based evaluation (**SWE-Bench Lite**).

**Project report:** [RACG-mini-SWE_report.pdf](report/RACG-mini-SWE_report.pdf)

## What we added
- **BM25 lexical retrieval**
- **Dense semantic retrieval** (embeddings + similarity search)
- **Hybrid retrieval** (BM25 + semantic reranking)
- **Initial trace-to-retrieve work**, which uses failing test outputs
  (stack traces, errors) as a localization signal  
  *(currently under development)*

## How to run retrieval experiments

```bash
python src/minisweagent/run/extra/swebench_rag_dense.py --help
python src/minisweagent/run/extra/swebench_rag_bm25_hybrid.py --help