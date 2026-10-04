import pytest
import torch
from transformers import AutoTokenizer

from streaming_shard_qwen import (
    TOKENIZER_NAME,
    LMStreamingDataset,
    ReadCounter,
    assign_files,
    collate_fn,
    rolling_token_blocks,
    skip_shard,
)

PAD_ID = 0


# ============================================================
# Rolling buffer -> fixed-length blocks
# ============================================================
def test_blocks_have_fixed_length():
    token_lists = [list(range(1, 50)), list(range(50, 300)), list(range(300, 310))]
    blocks = list(rolling_token_blocks(token_lists, block_size=128, pad_token_id=PAD_ID))
    assert all(b["input_ids"].shape == (128,) for b in blocks)
    assert all(b["attention_mask"].shape == (128,) for b in blocks)


def test_no_tokens_lost_or_reordered():
    token_lists = [list(range(1, 50)), list(range(50, 300)), list(range(300, 310))]
    blocks = list(rolling_token_blocks(token_lists, block_size=128, pad_token_id=PAD_ID))
    real_tokens = torch.cat([b["input_ids"][b["attention_mask"] == 1] for b in blocks]).tolist()
    assert real_tokens == list(range(1, 310))


def test_leftover_tokens_are_padded_and_masked():
    blocks = list(rolling_token_blocks([list(range(1, 11))], block_size=8, pad_token_id=PAD_ID))
    assert len(blocks) == 2
    last = blocks[-1]
    assert last["input_ids"].tolist() == [9, 10] + [PAD_ID] * 6
    assert last["attention_mask"].tolist() == [1, 1] + [0] * 6


def test_exact_multiple_produces_no_padding_block():
    blocks = list(rolling_token_blocks([list(range(16))], block_size=8, pad_token_id=PAD_ID))
    assert len(blocks) == 2
    assert all(b["attention_mask"].sum() == 8 for b in blocks)


# ============================================================
# Sharding
# ============================================================
@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
def test_skip_shard_partitions_examples(world_size):
    examples = [{"id": i} for i in range(103)]
    seen = []
    for rank in range(world_size):
        counter = ReadCounter()
        mine = [ex["id"] for ex in skip_shard(iter(examples), world_size, rank, counter)]
        assert counter.read == len(examples)  # every worker reads the whole stream
        assert counter.kept == len(mine)
        seen.extend(mine)
    assert sorted(seen) == list(range(103))  # each example used exactly once


@pytest.mark.parametrize("num_files,world_size", [(4, 4), (4, 2), (4, 1), (8, 4), (5, 2)])
def test_assign_files_partitions_files(num_files, world_size):
    files = [f"data/train-{i:05d}.parquet" for i in range(num_files)]
    assigned = [assign_files(files, rank, world_size) for rank in range(world_size)]
    flat = [f for part in assigned for f in part]
    assert sorted(flat) == files  # every file assigned exactly once
    sizes = [len(part) for part in assigned]
    assert max(sizes) - min(sizes) <= 1  # balanced across workers


def test_assign_files_more_workers_than_files():
    files = ["a.parquet", "b.parquet"]
    assigned = [assign_files(files, rank, 4) for rank in range(4)]
    assert assigned == [["a.parquet"], ["b.parquet"], [], []]


# ============================================================
# Batching
# ============================================================
def test_collate_shapes_and_labels():
    blocks = list(rolling_token_blocks([list(range(1, 1000))], block_size=128, pad_token_id=PAD_ID))[:8]
    batch = collate_fn(blocks)
    assert batch["input_ids"].shape == (8, 128)
    assert batch["attention_mask"].shape == (8, 128)
    assert torch.equal(batch["labels"], batch["input_ids"])
    assert batch["labels"].data_ptr() != batch["input_ids"].data_ptr()  # a copy, not the same tensor


# ============================================================
# End to end with the real Qwen2.5 tokenizer
# ============================================================
@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def test_eos_appended_after_every_story(tokenizer):
    stories = [{"text": "Once upon a time, there was a cat."},
               {"text": "The dog ran fast."},
               {"text": "They became friends."}]
    blocks = list(LMStreamingDataset(iter(stories), tokenizer, block_size=8))
    tokens = torch.cat([b["input_ids"][b["attention_mask"] == 1] for b in blocks]).tolist()
    assert tokens.count(tokenizer.eos_token_id) == len(stories)
    assert tokens[-1] == tokenizer.eos_token_id
    decoded = tokenizer.decode(tokens)
    assert decoded == "<|endoftext|>".join(s["text"] for s in stories) + "<|endoftext|>"
