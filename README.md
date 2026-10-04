# MLOps Lab 1 – LLM Data Pipeline

Lab submission for DADS7305 MLOps (Northeastern University), based on Session 3 – Data
(LLM data pipelines: tokenization, batching, sharding, streaming).

Builds a data pipeline that turns raw text into batches ready for causal language model training:
**text → tokens → fixed-length 128-token blocks → `[8, 128]` batches**, in memory, streamed, and sharded across worker processes.

## Source
The `original/` folder contains the unmodified lab files from the course repository:
[raminmohammadi/MLOps – Labs/Data_Labs/LLM_Data_Pipeline](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Data_Labs/LLM_Data_Pipeline)

## My modifications

| | Original lab | This submission |
|---|---|---|
| **Dataset** | WikiText-2 / AG News | **[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)** – 2.1M short stories in 4 parquet files |
| **Tokenizer** | GPT-2 / DistilBERT | **[Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B)** – modern BPE tokenizer, ~151k vocab |
| **Document boundaries** | Texts concatenated with no separator | `<|endoftext|>` appended after every story, so blocks show where stories end |
| **Sharding** | `manual_shard` only: every worker reads the whole stream and keeps every 4th example | Adds **file-level sharding** with Hugging Face `split_dataset_by_node` (each worker streams only its own file), with a `--mode` flag to compare both |
| **Sharding report** | Prints batch shapes only | Prints a per-worker summary: examples read vs. kept, % wasted reads, time, and the first tokens each worker saw |
| **Concatenation** | `sum(lists, [])` (quadratic) | `itertools.chain` (linear) |

## Files

| File | What it does |
|---|---|
| `Lab1.ipynb` | In-memory pipeline on the TinyStories validation split (21,990 stories → 35,646 blocks of 128 tokens) |
| `Lab2.ipynb` | Streaming pipeline over the full 2.1M-story train split with a rolling token buffer |
| `streaming_shard_qwen.py` | Streaming + 4-process sharding, comparing `manual` vs `by_node` sharding |
| `original/` | Unmodified course lab files |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Datasets and the tokenizer download automatically from the Hugging Face Hub on first run (no account needed).

## Running

```bash
jupyter notebook Lab1.ipynb      # or Lab2.ipynb

python streaming_shard_qwen.py --mode by_node   # file-level sharding
python streaming_shard_qwen.py --mode manual    # original approach, for comparison
```

Options: `--num-procs` (default 4), `--block-size` (128), `--batch-size` (8), `--batches` (3).

## Results

**Tokenizer** – Qwen2.5 on `"Once upon a time, there was a little cat."`:
```
['Once', 'Ġupon', 'Ġa', 'Ġtime', ',', 'Ġthere', 'Ġwas', 'Ġa', 'Ġlittle', 'Ġcat', '.']   (Ġ = leading space)
```

**Lab1** – TinyStories validation split: 4,563,985 tokens, 207.5 tokens per story on average,
packed into 35,646 training blocks; batches of shape `[8, 128]`.

**Sharding** – 4 workers, 3 batches each:

`--mode by_node` (each worker streams its own file):
```
rank |   read |   kept | wasted
   0 |     18 |     18 |     0%
   1 |     18 |     18 |     0%
   2 |     19 |     19 |     0%
   3 |     19 |     19 |     0%
```

`--mode manual` (original approach):
```
rank |   read |   kept | wasted
   0 |     73 |     19 |    74%
   1 |     70 |     18 |    74%
   2 |     67 |     17 |    75%
   3 |     76 |     19 |    75%
```

With `manual_shard`, every worker downloads and reads the whole stream and throws away about 3 of every 4 stories.
With `split_dataset_by_node`, TinyStories' 4 files are split evenly across 4 workers, so each worker reads only its own data.
This only works because the file count is divisible by the number of workers; otherwise `split_dataset_by_node`
falls back to the same skip-based approach as `manual_shard`.
