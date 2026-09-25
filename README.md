# Small Language Model (SLM) — Built From Scratch

A decoder-only GPT-style language model built entirely from scratch — architecture, tokenizer, training loop, and fine-tuning pipeline — as a personal learning project to understand how LLMs actually work under the hood, rather than just calling a library. Not intended for production use.

**Model weights (too large for GitHub) are hosted on Hugging Face:**
- Base pretrained model: [AkkiMaharaj/slm-30m-base](https://huggingface.co/AkkiMaharaj/slm-30m-base)
- Fine-tuned Q&A model: [AkkiMaharaj/slm-30m-qa](https://huggingface.co/AkkiMaharaj/slm-30m-qa)

---

## Architecture

Decoder-only GPT (~30M parameters):
- 8 layers, 8 attention heads
- 512 embedding/hidden dimension, 512-token context window
- Weight-tied input embedding and output head (saves ~8.4M parameters)
- PyTorch `scaled_dot_product_attention`, using FlashAttention automatically on compatible GPUs

---

## Phase 1: Data Preparation & Tokenization

- **Dataset:** English Wikipedia (`wikimedia/wikipedia`, `20231101.en`)
- **Tokenizer:** custom byte-level BPE, reduced vocabulary of 16,384 (keeps more of the parameter budget in transformer layers rather than embeddings)
- **RAM-safe streaming:** streamed via Hugging Face `datasets` with micro-batching (512 articles/batch) and reduced shuffle buffers to avoid OOM on Colab
- Tokenized into `uint16` memory-mapped binary shards (`train.bin`, `val.bin`) for efficient loading without holding the full dataset in RAM

## Phase 2: Base Pretraining

- **Optimizer:** AdamW (β1=0.9, β2=0.95, weight decay 0.1)
- **LR schedule:** cosine annealing with warmup (peak 1e-3, min 1e-4)
- **Stability:** gradient accumulation (8 steps) + gradient clipping (1.0)
- Periodic time-based checkpointing (every 15 minutes) for clean session restarts

## Phase 3: Instructional Fine-Tuning (Q&A)

- **Data:** Dolly-15k (filtered to short, direct categories), Alpaca-Cleaned (~4,000 sampled examples), and Wikipedia-grounded Q&A pairs extracted from lead sentences
- **Bug found & fixed:** tied embedding/output weights let the model learn a trivial "copy current token" shortcut when targets were misaligned; fixed with proper next-token target shifting and `-100` prompt masking
- **Checkpoint selection:** teacher-forced validation loss hides degeneration, so checkpoints were also evaluated on free-running generation against probe questions and only kept if outputs were non-repetitive and non-blank

## Phase 4: Inference (`generate.py`)

- Prompt templating (`### Question:\n...\n\n### Answer:\n`)
- Temperature scaling + top-k sampling
- Repetition penalty to reduce looping outputs
- Stops on `<|endoftext|>`

---

## Coming Next: 124M Parameter Model

Currently scaling this up to a 124M-parameter version, focused on understanding fine-tuning and training under constrained resources — writing the training/optimization approach myself rather than reusing an existing GPT-2 recreation script.

---

## Purpose

This project exists purely for learning: understanding tokenization, attention, pretraining, and instruction fine-tuning end-to-end. It is not a production model and shouldn't be treated as one.
