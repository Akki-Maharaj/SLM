"""
Generate text from your trained 30M Wikipedia GPT model.

By default this now talks to the fine-tuned Q&A checkpoint: your prompt is treated as a
question, wrapped in the same "### Question: / ### Answer:" template used for fine-tuning,
and generation stops automatically at <|endoftext|> instead of always running to max_new_tokens.

Usage:
    python generate.py --ckpt finetuned.pt --ask "Who is Joan of Arc?"
    python generate.py --ckpt finetuned.pt --ask "How did Stalin die?" --temperature 0.7
    python generate.py --ckpt finetuned.pt --ask "Summarize the French Revolution" --context "..."

    # old raw completion behavior (works with either checkpoint, no template applied):
    python generate.py --ckpt latest.pt --raw --prompt "The history of France"

Requires: torch, tokenizers  (pip install torch tokenizers)
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer


# ------------------------- model (must match training script exactly) -------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.n_embd, self.dropout = n_head, n_embd, dropout
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, repetition_penalty=1.0, stop_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :]

            # repetition penalty: discourage tokens already used in this sequence
            # (this is what fixes "Dictionary of Dictionary of Dictionary..." loops)
            if repetition_penalty != 1.0:
                for token_id in set(idx_cond[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            logits = logits / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

            # stop as soon as the model emits <|endoftext|> (only used in Q&A mode)
            if stop_id is not None and next_id.item() == stop_id:
                break
        return idx


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = GPT(cfg["vocab_size"], cfg["block_size"], cfg["n_layer"], cfg["n_head"], cfg["n_embd"], 0.0)
    model.load_state_dict(ck["model"])
    model.to(device)
    model.eval()
    print(f"loaded checkpoint from step {ck['step']} | config: {cfg}")
    return model


def build_qa_prompt(question, context=""):
    """Same template used during fine-tuning - must match exactly or answers degrade."""
    if context:
        return f"### Context:\n{context}\n\n### Question:\n{question}\n\n### Answer:\n"
    return f"### Question:\n{question}\n\n### Answer:\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="finetuned.pt",
                     help="checkpoint to load (finetuned.pt for Q&A, latest.pt for raw completion)")
    ap.add_argument("--tokenizer", default="tokenizer.json")

    # Q&A mode (default)
    ap.add_argument("--ask", default="", help="question to ask, wrapped in the Q&A template")
    ap.add_argument("--context", default="", help="optional context passage for the question")

    # raw completion mode (old behavior)
    ap.add_argument("--raw", action="store_true",
                     help="skip the Q&A template entirely and just continue --prompt as plain text, "
                          "like the original script did")
    ap.add_argument("--prompt", default="", help="raw text prompt, only used with --raw")

    ap.add_argument("--max_new_tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--rep_penalty", type=float, default=1.3)
    ap.add_argument("--n_samples", type=int, default=1)
    args = ap.parse_args()

    assert os.path.exists(args.ckpt), f"checkpoint not found: {args.ckpt}"
    assert os.path.exists(args.tokenizer), f"tokenizer not found: {args.tokenizer}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)
    tok = Tokenizer.from_file(args.tokenizer)
    eot_id = tok.token_to_id("<|endoftext|>")

    if args.raw:
        # ------------- old plain-completion behavior -------------
        prompt_text = args.prompt
        ids = tok.encode(prompt_text).ids if prompt_text else [0]
        stop_id = None
        echo_prefix_len = 0  # print the whole thing, prompt included, like before
    else:
        # ------------- Q&A mode -------------
        if not args.ask:
            ap.error("pass a question with --ask \"...\" (or use --raw --prompt \"...\" for plain completion)")
        prompt_text = build_qa_prompt(args.ask, args.context)
        ids = tok.encode(prompt_text).ids
        stop_id = eot_id
        echo_prefix_len = len(ids)  # only print the generated answer, not the template

    ctx = torch.tensor([ids], dtype=torch.long, device=device)

    if args.raw:
        print(f"\nprompt: {prompt_text!r}")
    else:
        print(f"\nquestion: {args.ask!r}" + (f"\ncontext: {args.context!r}" if args.context else ""))
    print(f"settings: temperature={args.temperature} top_k={args.top_k} repetition_penalty={args.rep_penalty}\n")

    for i in range(args.n_samples):
        out = model.generate(
            ctx.clone(), args.max_new_tokens, args.temperature, args.top_k, args.rep_penalty, stop_id=stop_id
        )
        out_ids = out[0].tolist()

        if not args.raw:
            gen_ids = out_ids[echo_prefix_len:]
            if gen_ids and gen_ids[-1] == eot_id:
                gen_ids = gen_ids[:-1]
            text = tok.decode(gen_ids)
        else:
            text = tok.decode(out_ids)

        print(f"--- sample {i + 1} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
