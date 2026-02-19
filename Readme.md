# Clinical Trial Outcome Prediction with LoRA Fine-Tuning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-clinical--trial--predictions-orange)](https://huggingface.co/datasets/3rdSon/clinical-trial-outcomes-predictions)

Train an AI to predict clinical trial outcomes with **73.3% accuracy** using automated dataset generation and LoRA fine-tuning. Built with [Lightning Rod Labs](https://www.lightningrod.ai/)' Future-as-Label methodology.

📝 **[Read the full article on Medium](MY_MEDIUM_LINK)** | 📊 **[View Dataset](https://huggingface.co/datasets/3rdSon/clinical-trial-outcomes-predictions)**

---

## Results

| Metric | Baseline | Fine-Tuned | Improvement |
|--------|----------|------------|-------------|
| **Accuracy** | 56.3% | **73.3%** | **+17.0pp** |
| **Correct** | 116/206 | 151/206 | +35 questions |

**Key Achievement:** 30% relative improvement over zero-shot baseline, matching expert-level performance (65-70%) on clinical trial predictions.
**[Pre-trained model available on Hugging Face](https://huggingface.co/3rdSon/clinical-trial-lora-llama3-8b)** - Use immediately without training!

---

## Quick Start

### Prerequisites

- Python 3.10+
- Lightning Rod API key ([get free credits](https://www.lightningrod.ai/))
- Hugging Face account
- **For fine-tuning:** GPU required (use [Google Colab](https://colab.research.google.com/) free tier)

### Installation

```bash
git clone https://github.com/3rdSon/clinical-trial-prediction-lora.git
cd clinical-trial-prediction-lora
pip install -r requirements.txt
```

### Running the Notebooks

**Follow in order:**

| Notebook | Description | Where to Run |
|----------|-------------|--------------|
| `01_dataset_generation.ipynb` | Generate dataset with Lightning Rod SDK | ✅ **Local (laptop)** |
| `02_baseline_evaluation.ipynb` | Zero-shot evaluation | ⚠️ **Google Colab (GPU)** |
| `03_model_finetuning.ipynb` | LoRA fine-tuning (~21 min) | ⚠️ **Google Colab (GPU)** |
| `04_results_analysis.ipynb` | Create visualizations | ✅ **Local or Colab** |

**⚠️ GPU Required:** Notebooks 02-03 need GPU. Use [Google Colab](https://colab.research.google.com/) with T4 runtime (free). Don't run on laptop unless you have a dedicated GPU.

**✅ No GPU Needed:** Notebook 01 (dataset generation) runs fine on your laptop - takes ~2 minutes.

**🚀 Skip Training?** Download the [pre-trained model from Hugging Face](https://huggingface.co/3rdSon/clinical-trial-lora-llama3-8b) and start making predictions immediately!

---

## 📁 Repository Structure

```
clinical-trial-prediction-lora/
├── notebooks/              # Jupyter notebooks (run in order)
│   ├── 01_dataset_generation.ipynb
│   ├── 02_baseline_evaluation.ipynb
│   ├── 03_model_finetuning.ipynb
│   └── 04_results_analysis.ipynb
├── data/                   # Generated datasets
│   ├── clinical_data.csv           # Full valid dataset (1,366 examples)
│   ├── clinical_train.csv          # Training set (85%)
│   ├── clinical_test.csv           # Test set (15%)
│   ├── baseline_results.csv        # Baseline predictions
│   ├── finetuned_results.csv       # Fine-tuned predictions
│   └── comparison_results.csv      # Side-by-side comparison
├── results/                # Visualizations and analysis
│   ├── accuracy_comparison.png
│   ├── confusion_matrix_comparison.png
│   ├── improvement_breakdown.png
│   └── key_insights.md
├── models/                 # Model storage info
│   └── README.md
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## How It Works

### 1. Automated Dataset Generation
Uses Lightning Rod's Future-as-Label: historical questions (2023-2024) are automatically labeled using verified outcomes (late 2024/2025). No manual labeling required!

- **Generated:** 1,882 questions in ~10 minutes
- **Valid:** 1,366 high-confidence labels (72.6%)
- **Source:** Clinical trial news from 2023-2024

### 2. LoRA Fine-Tuning
- **Model:** Llama-3-8B with 4-bit quantization
- **Method:** Low-Rank Adaptation (only 0.2% of parameters trainable)
- **Training:** 3 epochs, ~35 minutes on free Colab T4 GPU
- **Result:** 73.3% accuracy (vs 56.3% baseline)

### 3. Key Patterns Learned
✅ Company track records matter (big pharma vs startups)  
✅ Therapeutic area success rates (metabolic 68% vs oncology 48%)  
✅ Timeline realism (spotted unrealistic trial schedules)  
✅ Better failure prediction (fixed 59 baseline errors)

Full analysis: [results/key_insights.md](results/key_insights.md)

---

## Dataset

**🤗 [View on Hugging Face](https://huggingface.co/datasets/3rdSon/clinical-trial-outcomes-predictions)**

- 1,366 clinical trial predictions (2023-2024)
- Binary labels with 99.8% average confidence
- Full provenance (source articles + outcome verification)

```python
from datasets import load_dataset
dataset = load_dataset("3rdSon/clinical-trial-outcomes-predictions")
```

---

## Repository Structure

```
clinical-trial-prediction-lora/
├── notebooks/           # Run these in order (01 → 04)
├── data/               # Generated datasets and results
├── results/            # Visualizations and analysis
├── models/             # Model info (see models/README.md)
└── requirements.txt    # Python dependencies
```

---

## Resources

### This Project
- 🤖 [Model on Hugging Face](https://huggingface.co/3rdSon/clinical-trial-lora-llama3-8b)
- 📊 [Dataset on Hugging Face](https://huggingface.co/datasets/3rdSon/clinical-trial-outcomes-predictions)
- 📝 [Full Article on Medium](MY_MEDIUM_LINK)
- 💻 [Code on GitHub](https://github.com/3rd-Son/clinical-trial-prediction-lora)

### Lightning Rod Labs
- [Website](https://www.lightningrod.ai/) | 📦 [Python SDK](https://github.com/lightning-rod-labs/lightningrod-python-sdk) | 🤗 [Hugging Face](https://huggingface.co/LightningRodLabs)

### Tools Used
- [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) - Base model
- [Unsloth](https://github.com/unslothai/unsloth) - Efficient LoRA training
- [Google Colab](https://colab.research.google.com/) - Free GPU

---

## Applications

This methodology works for any temporal prediction task with historical data:

- Product launches, policy outcomes, market events, sports predictions
- Requirements: ✅ Historical data ✅ Verifiable outcomes ✅ Sufficient examples

---

## Limitations

- Limited to 2022-2024 data; less reliable for novel mechanisms
- **Not medical advice** - for research only
- 73% accuracy is meaningful but not prophecy

---

## License

MIT License | Dataset: Apache 2.0

---

## Acknowledgments

Built with [Lightning Rod Labs](https://www.lightningrod.ai/)' Future-as-Label methodology. Thanks to Unsloth and Meta AI.

---

## Contact

**Victory Nnaji**  
GitHub: [@3rdSon](https://github.com/3rd-Son) | Hugging Face: [@3rdSon](https://huggingface.co/3rdSon) | Medium: [YOUR_MEDIUM_PROFILE] | LinkedIn : [@Victory Nnaji](https://www.linkedin.com/in/3rdson/) 

---

## Citation

```bibtex
@misc{nnaji2025clinical_trial_prediction,
  author = {Victory},
  title = {Clinical Trial Outcome Prediction with LoRA Fine-Tuning},
  year = {2025},
  howpublished = {\url{https://github.com/3rdSon/clinical-trial-prediction-lora}}
}
```

