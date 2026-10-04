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
| **Sharding** | `manual_shard` only: every worker reads the whole stream and keeps every 4th example (splits *after* reading) | **File-level sharding that splits *before* reading**: hand-written `manual` mode lists the dataset's 4 files and gives each worker its own, plus a `by_node` mode using Hugging Face `split_dataset_by_node`. The original approach is kept as `skip` mode for comparison |
| **Sharding report** | Prints batch shapes only | Prints a per-worker summary: examples read vs. kept, % wasted reads, time, and the first tokens each worker saw |
| **Concatenation** | `sum(lists, [])` (quadratic) | `itertools.chain` (linear) |

## Files

| File | What it does |
|---|---|
| `Lab1.ipynb` | In-memory pipeline on the TinyStories validation split (21,990 stories → 35,646 blocks of 128 tokens) |
| `Lab2.ipynb` | Streaming pipeline over the full 2.1M-story train split with a rolling token buffer |
| `streaming_shard_qwen.py` | Streaming + 4-process sharding with three modes: `manual`, `by_node`, `skip` |
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

python streaming_shard_qwen.py --mode manual    # our file-level sharding (default)
python streaming_shard_qwen.py --mode by_node   # same idea via Hugging Face split_dataset_by_node
python streaming_shard_qwen.py --mode skip      # original lab approach, for comparison
```

| Mode | When the split happens | How |
|---|---|---|
| `manual` | **Before reading** | List the train files (`data/train-0000X-of-00004-*.parquet`), give worker `r` files `r, r+N, r+2N, …`, and stream only those |
| `by_node` | **Before reading** | `split_dataset_by_node(ds, rank, world_size)` assigns files to workers for us |
| `skip` | After reading | Original `manual_shard`: every worker reads every story and keeps `idx % N == rank` |

Options: `--num-procs` (default 4), `--block-size` (128), `--batch-size` (8), `--batches` (3).

## Results

**Tokenizer** – Qwen2.5 on `"Once upon a time, there was a little cat."`:
```
['Once', 'Ġupon', 'Ġa', 'Ġtime', ',', 'Ġthere', 'Ġwas', 'Ġa', 'Ġlittle', 'Ġcat', '.']   (Ġ = leading space)
```

**Lab1** – TinyStories validation split: 4,563,985 tokens, 207.5 tokens per story on average,
packed into 35,646 training blocks; batches of shape `[8, 128]`.

**Sharding** – 4 workers, 3 batches each:

`--mode manual` (our file-level split, before reading):
```
[rank 0] assigned files: ['train-00000-of-00004']
[rank 1] assigned files: ['train-00001-of-00004']
[rank 2] assigned files: ['train-00002-of-00004']
[rank 3] assigned files: ['train-00003-of-00004']

rank |   read |   kept | wasted | first tokens
   0 |     18 |     18 |     0% | 'One day, a little girl named Lily found a needle in'
   1 |     18 |     18 |     0% | 'Once upon a time, there lived a very loud mushroom.'
   2 |     19 |     19 |     0% | 'Once upon a time, there was a high sun in the'
   3 |     19 |     19 |     0% | 'Once upon a time, there was a boy named Tim.'
```

`--mode by_node` gives the identical split (same counts, same first story per worker), which confirms the
hand-written version matches what `split_dataset_by_node` does internally.

`--mode skip` (original approach, split after reading):
```
rank |   read |   kept | wasted
   0 |     73 |     19 |    74%
   1 |     70 |     18 |    74%
   2 |     67 |     17 |    75%
   3 |     76 |     19 |    75%
```

With the original `manual_shard`, every worker downloads and reads the whole stream and throws away about 3 of every 4 stories.
Splitting by file before reading means each worker reads only its own data.
File-level splitting is limited by the number of files: TinyStories has 4, so with more than 4 workers the
extra workers get no files (`manual` mode prints a message and skips them), and `split_dataset_by_node`
falls back to the skip-based approach when the file count isn't divisible by the number of workers.
