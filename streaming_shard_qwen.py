# streaming_shard_qwen.py
# Streaming + multi-process sharding for causal LM training data.
# Dataset: TinyStories (4 parquet files)  |  Tokenizer: Qwen2.5
#
# Compares two sharding strategies:
#   manual  - every worker reads the WHOLE stream and keeps every Nth example (original lab)
#   by_node - Hugging Face split_dataset_by_node: each worker reads only ITS OWN file(s)
#
# Usage:
#   python streaming_shard_qwen.py --mode by_node
#   python streaming_shard_qwen.py --mode manual
import argparse
import os
import time
from datetime import datetime

import torch
import multiprocessing as mp
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from transformers import AutoTokenizer

DATASET_NAME = "roneneldan/TinyStories"
TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B"


# ============================================================
# Counter so we can see how much of the stream each worker reads
# ============================================================
class ReadCounter:
    def __init__(self):
        self.read = 0   # examples pulled from the stream
        self.kept = 0   # examples this worker actually uses


# ============================================================
# Sharding strategies
# ============================================================
def manual_shard(dataset_iter, num_shards, process_index, counter):
    # Original lab approach: read everything, keep idx % num_shards == process_index
    for idx, example in enumerate(dataset_iter):
        counter.read += 1
        if idx % num_shards == process_index:
            counter.kept += 1
            yield example


def node_shard(dataset, rank, world_size, counter):
    # File-level sharding: with 4 files and 4 workers, each worker streams exactly one file
    for example in split_dataset_by_node(dataset, rank=rank, world_size=world_size):
        counter.read += 1
        counter.kept += 1
        yield example


# ============================================================
# Rolling buffer: tokens -> fixed-length blocks
# ============================================================
def rolling_token_blocks(token_iter, block_size, pad_token_id):
    buffer = []
    for tokens in token_iter:
        buffer.extend(tokens)
        while len(buffer) >= block_size:
            chunk = buffer[:block_size]
            buffer = buffer[block_size:]
            yield {
                "input_ids": torch.tensor(chunk, dtype=torch.long),
                "attention_mask": torch.ones(block_size, dtype=torch.long),
            }
    # Pad leftover tokens if any remain
    if buffer:
        padded = buffer + [pad_token_id] * (block_size - len(buffer))
        yield {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(buffer) + [0] * (block_size - len(buffer)), dtype=torch.long),
        }


class LMStreamingDataset(IterableDataset):
    def __init__(self, example_iter, tokenizer, block_size):
        self.example_iter = example_iter
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        # Append <|endoftext|> after every story so the model sees document boundaries
        eos = self.tokenizer.eos_token_id
        token_stream = (
            self.tokenizer(ex["text"], add_special_tokens=False)["input_ids"] + [eos]
            for ex in self.example_iter
        )
        yield from rolling_token_blocks(token_stream, self.block_size, self.tokenizer.pad_token_id)


def collate_fn(batch):
    input_ids = torch.stack([ex["input_ids"] for ex in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": torch.stack([ex["attention_mask"] for ex in batch]),
        "labels": input_ids.clone(),
    }


# ============================================================
# Worker
# ============================================================
def worker_entry(rank, world_size, mode, block_size, batch_size, batches_to_show, results):
    stream_ds = load_dataset(DATASET_NAME, split="train", streaming=True)
    counter = ReadCounter()

    if mode == "manual":
        example_iter = manual_shard(stream_ds, world_size, rank, counter)
    else:
        example_iter = node_shard(stream_ds, rank, world_size, counter)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    lm_dataset = LMStreamingDataset(example_iter, tokenizer, block_size)
    loader = DataLoader(lm_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0)

    print(f"{datetime.now():%H:%M:%S} [PID {os.getpid()} | rank {rank}] starting ({mode})", flush=True)
    first_tokens = None
    start = time.time()
    for i, batch in enumerate(loader):
        if first_tokens is None:
            first_tokens = batch["input_ids"][0, :12]
        print(f"{datetime.now():%H:%M:%S} [rank {rank}] batch {i} -> {tuple(batch['input_ids'].shape)}", flush=True)
        if i + 1 >= batches_to_show:
            break
    elapsed = time.time() - start

    preview = tokenizer.decode(first_tokens).replace("\n", " ")
    results[rank] = (counter.read, counter.kept, elapsed, preview)
    print(f"{datetime.now():%H:%M:%S} [rank {rank}] done.", flush=True)


# ============================================================
# Launcher
# ============================================================
def launch_multi_proc(num_procs, mode, block_size, batch_size, batches_to_show):
    ctx = mp.get_context("spawn")  # Safe for all platforms
    manager = ctx.Manager()
    results = manager.dict()
    procs = []
    for rank in range(num_procs):
        p = ctx.Process(target=worker_entry,
                        args=(rank, num_procs, mode, block_size, batch_size, batches_to_show, results))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    print(f"\nSummary - mode={mode}, workers={num_procs}, block_size={block_size}, batch_size={batch_size}")
    print(f"{'rank':>4} | {'read':>6} | {'kept':>6} | {'wasted':>6} | {'secs':>5} | first tokens")
    for rank in sorted(results.keys()):
        read, kept, secs, preview = results[rank]
        wasted = f"{(read - kept) / read:.0%}" if read else "-"
        print(f"{rank:>4} | {read:>6} | {kept:>6} | {wasted:>6} | {secs:>5.1f} | {preview!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming + sharding demo (TinyStories / Qwen2.5)")
    parser.add_argument("--mode", choices=["manual", "by_node"], default="by_node")
    parser.add_argument("--num-procs", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=3)
    args = parser.parse_args()

    launch_multi_proc(num_procs=args.num_procs,
                      mode=args.mode,
                      block_size=args.block_size,
                      batch_size=args.batch_size,
                      batches_to_show=args.batches)
