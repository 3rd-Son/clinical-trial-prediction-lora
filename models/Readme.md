# Model Storage

## Fine-Tuned Model

The fine-tuned LoRA adapter is **available on Hugging Face** for immediate use!

🤗 **[3rdSon/clinical-trial-lora-llama3-8b](https://huggingface.co/3rdSon/clinical-trial-lora-llama3-8b)**

---

## Quick Start

### Load the Model Directly

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="3rdSon/clinical-trial-lora-llama3-8b",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)

# Enable inference mode
FastLanguageModel.for_inference(model)
```

### Make a Prediction

```python
prompt = """You are evaluating clinical trial outcomes. Based on the question below, predict whether the outcome will be YES (1) or NO (0).

Question: Will Eli Lilly's Phase 3 obesity drug trial meet primary endpoints by Q4 2025?

Respond with only a single digit: 0 or 1.
Answer:"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=10)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Alternative: Train It Yourself

If you prefer to train from scratch:

1. Run `notebooks/01_dataset_generation.ipynb`
2. Run `notebooks/02_baseline_evaluation.ipynb`
3. Run `notebooks/03_model_finetuning.ipynb` (requires GPU - use Google Colab)

Training takes ~21 minutes on a free Google Colab T4 GPU.

---

## Model Details

### Fine-Tuned Model Specifications

- **Base Model:** Llama-3-8B
- **Method:** LoRA (Low-Rank Adaptation)
- **Trainable Parameters:** ~16M (0.2% of total)
- **LoRA Configuration:**
  - Rank (r): 16
  - Alpha: 16
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Training Details

- **Dataset:** [clinical-trial-outcomes-predictions](https://huggingface.co/datasets/3rdSon/clinical-trial-outcomes-predictions)
- **Training samples:** 1,161 clinical trial predictions (2023-2024)
- **Epochs:** 3
- **Batch size:** 2 (effective 8 with gradient accumulation)
- **Learning rate:** 2e-4
- **Optimizer:** AdamW 8-bit
- **Hardware:** Google Colab T4 GPU (free tier)
- **Training time:** ~35 minutes

### Performance

| Metric | Baseline (Zero-Shot) | Fine-Tuned (LoRA) | Improvement |
|--------|---------------------|-------------------|-------------|
| **Accuracy** | 56.3% | **73.3%** | **+17.0pp** |
| **Correct Predictions** | 116/206 | 151/206 | +35 questions |
| **Relative Improvement** | - | - | **+30.2%** |

The fine-tuned model achieves accuracy comparable to pharmaceutical domain experts (65-70%).

---

## Base Model

The base Llama-3-8B model is automatically downloaded from Hugging Face when you load the fine-tuned version.

- **Model:** `meta-llama/Meta-Llama-3-8B`
- **Requires:** Hugging Face account + access approval (free, instant)
- **Request access:** [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

---

## Storage Options

### Option 1: Load from Hugging Face (Recommended)
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="3rdSon/clinical-trial-lora-llama3-8b",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)
```

### Option 2: Download Locally
```bash
# Using huggingface-cli
huggingface-cli download 3rdSon/clinical-trial-lora-llama3-8b

# Then load locally
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./clinical-trial-lora-llama3-8b",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)
```

### Option 3: Train and Save to Google Drive
If you train the model yourself on Colab:

```python
# After training, backup to Google Drive
from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copytree(
    "clinical_trial_lora",
    "/content/drive/MyDrive/clinical_trial_lora",
    dirs_exist_ok=True
)

# Later, load from Drive
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/content/drive/MyDrive/clinical_trial_lora",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)
```

---

## Model Size

- **LoRA Adapter:** ~500MB
- **Full Model (with base):** ~15GB (when loaded in 4-bit)
- **Memory Requirements:** ~6GB GPU RAM (4-bit quantization)

---

## Links

- 🤗 **Model:** [Hugging Face](https://huggingface.co/3rdSon/clinical-trial-lora-llama3-8b)
- 📊 **Dataset:** [Hugging Face](https://huggingface.co/datasets/3rdSon/clinical-trial-outcomes-predictions)
- 💻 **Code:** [GitHub](https://github.com/3rd-Son/clinical-trial-prediction-lora)
- 📝 **Article:** [Medium](YOUR_MEDIUM_LINK)

---

## Questions?

For issues related to model access, loading, or inference, please open an issue in the [main repository](https://github.com/3rd-Son/clinical-trial-prediction-lora/issues).