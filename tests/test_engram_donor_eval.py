import torch

from scripts.engram_donor_eval import build_engram_input_ids, compute_token_ranks


class DummyTokenizer:
    def encode(self, text):
        return [ord(ch) for ch in text]


def test_build_engram_input_ids_uses_donor_aligned_stream():
    tokenizer = DummyTokenizer()
    prompt_ids = [10, 11, 12]
    donor_text = "abcd"
    out = build_engram_input_ids(prompt_ids, donor_text, tokenizer, max_seq_len=5)
    assert out == [97, 98, 99]


def test_compute_token_ranks_descending():
    logits = torch.tensor([0.1, 2.0, -1.0, 1.5])
    ranks = compute_token_ranks(logits)
    assert ranks.tolist() == [3, 1, 4, 2]
