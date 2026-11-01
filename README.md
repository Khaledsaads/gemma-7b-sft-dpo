# Gemma 7B: Supervised Fine-Tuning (SFT) & Direct Preference Optimization (DPO)

A parameter-efficient fine-tuning project exploring supervised instruction tuning followed by preference optimization using **QLoRA** and the Gemma 7B language model.

## Overview

This project implements a two-stage fine-tuning pipeline:

1. **Supervised Fine-Tuning (SFT):** Fine-tune Gemma 7B on instruction-response examples to adapt its responses to the training data.
2. **Direct Preference Optimization (DPO):** Further optimize the SFT model using preference pairs containing chosen and rejected responses.

The project explores how supervised instruction tuning and preference optimization affect the quality of generated responses.

## Model and Datasets
### Base Model

- **Model:** [`unsloth/gemma-7b-bnb-4bit`](https://huggingface.co/unsloth/gemma-7b-bnb-4bit)
- **Architecture:** Gemma 7B
- **Quantization:** 4-bit

### Training Datasets

| Stage | Dataset | Purpose |
|---|---|---|
| SFT | [`HuggingFaceH4/deita-10k-v0-sft`](https://huggingface.co/datasets/HuggingFaceH4/deita-10k-v0-sft) | Supervised instruction tuning |
| DPO | [`argilla/dpo-mix-7k`](https://huggingface.co/datasets/argilla/dpo-mix-7k) | Preference-based optimization |

## Methodology

### 1. Supervised Fine-Tuning (SFT)

The quantized Gemma 7B model is fine-tuned on the Deita SFT dataset using LoRA adapters.

The training process updates a small subset of trainable adapter parameters rather than the full model weights, reducing memory requirements.

### 2. Direct Preference Optimization (DPO)

The SFT model is subsequently fine-tuned using the DPO preference dataset.

DPO uses chosen and rejected responses to encourage the model to assign relatively higher preference to desirable responses. This stage builds on the behavior learned during SFT.

### Parameter-Efficient Fine-Tuning Configuration

| Parameter | Value |
|---|---|
| Base model | `unsloth/gemma-7b-bnb-4bit` |
| Fine-tuning method | QLoRA |
| Quantization | 4-bit |
| LoRA rank (`r`) | 16 |
| Training epochs | 2 |
| Maximum sequence length | 2048 |
| Trainable parameters | 50,003,968 |
| Total parameters | 8,587,684,864 |
| Trainable parameter ratio | Approximately 0.58% |


## Hardware and Training

- **SFT and DPO training:** NVIDIA A100 GPU
- **Additional experiments and inference:** NVIDIA T4 GPU

The project leverages Unsloth, quantization, and LoRA adapters to enable memory-efficient fine-tuning of a 7B-parameter model.
## DPO Training Results

The following values are from the recorded DPO training run.
| Step | Training Loss | Validation Loss | Rewards/accuracies | Rewards/margins |
| ---- | ------------- | --------------- | ------------------ | --------------- |
| 0    | ~= 0.693      | ~= 0.693        | —                  | —               |
| 80   | 0.908158      | 0.672103        | 0.645312           | 1.116044        |
| 160  | 0.961590      | 0.665290        | 0.709375           | 3.680988        |
| 240  | 0.675935      | 0.711836        | 0.701563           | 1.960260        |
| 320  | 0.781759      | 0.664966        | 0.673438           | 1.450833        |
| 400  | 0.913089      | 0.678688        | 0.704688           | 2.754630        |
| 480  | 0.406533      | 0.627990        | 0.885938           | 6.750652        |
| 560  | 0.631789      | 0.582193        | 0.728125           | 1.655571        |
| 640  | 0.502086      | 0.540391        | 0.787500           | 2.413145        |
| 720  | 0.571136      | 0.460994        | 0.760938           | 2.025602        |
| 800  | 0.480923      | 0.465995        | 0.789063           | 2.188343        |
| 840  | 0.491203      | 0.455060        | 0.779014           | 2.172718        |

The validation loss decreased across the reported checkpoints, reaching **0.455060 at step 840**. These metrics describe the recorded training run and do not, by themselves, establish that generated responses improved in quality.

### Learning Rate Schedule

The DPO learning rate followed a scheduled decay, reaching approximately `7.71e-10` at step 840.

The learning rate and loss curves should be interpreted together when evaluating optimization behavior.

## Model Evaluation

### Base vs. SFT vs. DPO

The three model versions will be evaluated using the same prompts to compare their generated responses.

# SFT vs. DPO Eval

Comparison of `assistant_sft` vs `assistant_dpo` on 50 prompts across **correctness**, **relevance**, **instruction following**, and **clarity**.

---

## Setup

Both variants must be generated with **identical** decoding params:
| Param | Value |
|---|---|
| Temperature | `0.7` |
| Top-p | `0.9` |
| Top-k | `50` |
| Max new tokens | `None` |
| Seed | same |
| Prompt template | identical |

**Evaluation plan**

- Use identical prompts across all three model versions.
- Compare correctness, relevance, instruction following, and clarity.
- Record failure cases and factual inaccuracies, not just successful examples.
- Keep decoding parameters consistent to make the comparison fair.


**Rule:** correctness beats style. A clear but wrong answer loses.

---

## Results

| Outcome | Count | Share |
|---|---:|---:|
| DPO preferred | 21 | 42% |
| SFT preferred | 10 | 20% |
| Tie | 18 | 36% |
| Both fail | 1 | 2% |

**DPO wins** on explanation-heavy prompts (Gutenberg, process vs. thread, OSI, DNS, mitochondria).
**SFT wins** on short technical prompts (JS class toggle, REST methods, `const`, email regex, deadlock, `var`).
**DPO risk:** verbosity hides errors. **SFT risk:** brevity omits edge cases.

---

## Failure Cases

| ID | Variant | Issue | Sev. |
|---:|---|---|---|
| 2 | SFT | Compound interest `$157.41`; correct is `$157.63` | High |
| 1 | SFT | Palindrome strips only selected punctuation | Med |
| 10 | SFT | Full-moon wording inaccurate | Med |
| 10 | DPO | Moon phase order wrong; libration claims wrong | High |
| 12 | DPO | "Transferful"; DELETE body claim; CONNECT listed | High |
| 13 | DPO | Says `Const` is a JS keyword | Med |
| 19 | DPO | Photosynthesis oversimplified (skips ATP/NADPH) | Med |
| 22 | DPO | Regex unescaped `.`, no anchors | High |
| 22 | SFT | Regex too simplified (TLD, special chars) | Med |
| 24 | DPO | Says index keys are always unique | Med |
| 26 | DPO | Says `netstat` shows closed ports | Med |
| 29 | SFT | Overstates asymmetric MITM resistance | Med |
| 36 | SFT | Factorial returns `1` for negatives | Med |
| 37 | DPO | Lists Coffman conditions as avoidance principles | High |
| 38 | DPO | `var` example claims `undefined`; prints function | High |
| 9 | DPO | Adds helper fn; violates "short snippet" | Low |
| 25 | DPO | Adds extra "pillars" to OOP core four | Low |

---

## Reproduction

1. Load JSON.
2. Generate both variants with identical decoding params.
3. Score on the 4 criteria.
4. Log every failure, including minor ones.
5. Prefer correctness over style.

---

## Notes

- "Tie" = both acceptable, similar quality (not identical).
- Decoding table is a template — fill in actual values per run.



## Qualitative Testing

Initial generation tests covered programming, mathematics, general knowledge, machine learning, SQL, and everyday explanations.

The examples revealed both useful responses and limitations, including:

- Correct basic calculations and several straightforward programming answers.
- A case-sensitivity issue in a palindrome implementation.
- Incorrect explanations of some lunar phases.
- An inaccurate expansion of the term REST.
- Opportunities to improve precision and reduce unnecessary verbosity.

These observations are preliminary qualitative findings. They are not a formal benchmark and should not be interpreted as evidence that DPO outperforms the base or SFT model.

## Limitations and Future Work

- Complete the controlled Base vs. SFT vs. DPO comparison.
- Evaluate more prompts across coding, reasoning, factual accuracy, and instruction following.
- Investigate failure cases and measure whether preference optimization reduces them.
- Add quantitative evaluation metrics where appropriate.
- Document the final adapter checkpoints and inference procedure.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- TRL
- PEFT / LoRA
- BitsAndBytes 4-bit quantization
- Unsloth

## Repository Structure

```text
.
├── notebooks/
│   ├── sft_training.ipynb
│   ├── dpo_training.ipynb
│   └── model_evaluation.ipynb
├── evaluation/
│   └── test_cases.json
├── README.md
└── requirements.txt
```