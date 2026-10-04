# MLOps Lab 1 – LLM Data Pipeline

Lab submission for DADS7305 MLOps (Northeastern University), based on Session 3 – Data
(LLM data pipelines: tokenization, batching, sharding, streaming).

## Source
The `original/` folder contains the unmodified lab files from the course repository:
[raminmohammadi/MLOps – Labs/Data_Labs/LLM_Data_Pipeline](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Data_Labs/LLM_Data_Pipeline)

- `Lab1.ipynb` – in-memory pipeline: WikiText-2 → GPT-2 tokenizer → 128-token blocks → DataLoader
- `Lab2.ipynb` – streaming pipeline with a rolling token buffer
- `streaming_shard.py` – streaming + multi-process sharding (AG News, DistilBERT)
- `streaming_shard_gpt2.py` – streaming + multi-process sharding for causal LM (WikiText, GPT-2)

## My modifications
_Work in progress._
