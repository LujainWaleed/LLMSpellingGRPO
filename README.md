# LLMSpellingGRPO

## Project Overview

This project demonstrates a complete fine-tuning pipeline on a simple but illustrative task: teaching a language model to split words letter by letter.

## Key Concepts

| Concept | Description |
|---|---|
| **LoRA** (Low-Rank Adaptation) | Adds small trainable layers on top of a frozen model — reduces trainable parameters from 134M to ~3.6M (2.67%) |
| **SFT** (Supervised Fine-Tuning) | Trains the model by showing it correct input-output pairs |
| **GRPO** (Group Relative Policy Optimization) | Trains the model using reward functions — no labeled answers needed, just a scoring function |
| **Reward Functions** | Two functions used: one scores character accuracy, another checks correct hyphen formatting |
| **Overfitting** | Model performs better on training data than unseen test data — a known limitation with small datasets |

---

## Results

| Stage | Train Score | Test Score |
|---|---|---|
| Base Model (no training) | Poor | Poor |
| After SFT | 13.51 / 20 | 2.5583 / 7|
| After GRPO | 13.97 / 20 | 2.64 / 7 |

The gap between train and test scores indicates overfitting due to the small dataset size (61 words total).

---

## Dataset

- 61 English words of varying lengths (4–8 letters)
- Each word is turned into a record with:
  - `prompt` — instruction asking the model to spell the word
  - `completion` — correct hyphenated spelling (e.g., `T-R-I-U-M-P-H.`)
  - `word` — original word
  - `spelling` — letter by letter spelling without trailing period
- Split: **90% train / 10% test**

---


## Requirements

- [Transformers](https://huggingface.co/docs/transformers) — model loading
- [TRL](https://huggingface.co/docs/trl) — SFTTrainer and GRPOTrainer
- [PEFT](https://huggingface.co/docs/peft) — LoRA configuration
- [Datasets](https://huggingface.co/docs/datasets) — dataset creation and splitting
- [PyTorch](https://pytorch.org/) — model training backend
