# audio-llm-ser-evaluation
> Systematic Evaluation & Benchmarking of Large Audio-Language Models (LALMs) on Speech Emotion Recognition (SER).

---

## 🎯 Overview

`audio-llm-ser-evaluation` is a modular, reproducible experimentation framework designed to benchmark and analyze how modern Large Audio-Language Models (LALMs) perceive, reason about, and classify human emotion from speech.

Rather than focusing solely on fine-tuning, this repository provides a comprehensive evaluation pipeline covering:
- **Zero-Shot & Prompting Paradigms:** Zero-shot, Chain-of-Thought (CoT), Structured CoT-JSON, and Multi-turn context.
- **Model Architectures & Quantization:** Evaluating native audio-LLMs across precision modes (e.g., `BF16`, `NF4`).
- **Multimodal & Acoustic Interventions:** Assessing the impact of conversational context, acoustic vs. textual bias, and domain transfer.
- **Rigorous Evaluation:** Standardized metrics (Macro-F1, Weighted-F1, Per-Class F1, Confusion Matrices, Statistical Significance).

---

## 🏗️ Repository Architecture

The project is structured around modular, composable configurations to prevent code duplication and guarantee experiment reproducibility:

```
audio-llm-ser-evaluation/
├── configs/                       # Composable experiment configurations
│   ├── base.yaml                  # Global defaults (seeds, tracking, metrics)
│   ├── datasets/                  # Dataset paths, splits, and label definitions
│   │   ├── meld.yaml
│   │   └── iemocap.yaml
│   ├── models/                    # Model identifiers, precision, generation parameters
│   │   ├── audio-flamingo.yaml
│   │   ├── qwen-omni.yaml
│   │   └── whisper-mistral.yaml
│   ├── prompts/                   # Prompt strategies and instruction templates
│   │   ├── zero-shot.yaml
│   │   ├── cot.yaml
│   │   ├── cot-json.yaml
│   │   └── multi-turn.yaml
│   └── experiments/
│       └── experiments.yaml       # Matrix definitions and ablation sweeps
├── data/                          # Dataset storage (downloaded & preprocessed)
├── results/                       # Standardized outputs, metrics, and confusion matrices
├── scripts/                       # Orchestration scripts
└── src/
    ├── data_acquisition/          # Automated dataset download & extraction
    ├── preprocessing/             # Audio normalization (16kHz mono), context formatting
    ├── models/                    # Model wrappers and inference adapters
    ├── prompting/                 # Dynamic prompt templating
    ├── inference/                 # Batching and matrix runners
    ├── evaluation/                # Metrics, confusion matrices, error analysis
    ├── tracking/                  # Artifact & metadata logging
    └── utils/                     # Emotion maps and configuration loaders
```

---

## 📊 Datasets

| Dataset | Modality | Classes | Focus |
|---|---|---|---|
| **MELD** | Audio + Transcript | 7 emotions (*anger, disgust, fear, joy, neutral, sadness, surprise*) | Multi-party dialogue in TV show setting |
| **IEMOCAP** | Audio + Transcript | 8/4 emotions (*angry, excited, fear, happy, neutral, sad, surprise, frustration*) | Dyadic improvised & scripted emotional interactions |

---

## 🤖 Supported Models

- **Audio Flamingo 3 (`af3`)**: Multimodal model optimized for audio understanding.
- **Qwen-Omni (`qwen_omni`)**: End-to-end speech and language conversational model.
- **Whisper-Mistral (`wm7`)**: Cascaded/integrated speech transcription and reasoning architecture.

---

## 🧪 Experiment Matrix & Prompting Strategies

Experiments are declared declaratively in `configs/experiments/experiments.yaml`. Key benchmark tracks include:

1. **Main Benchmark (`main`)**: Comprehensive evaluation across models, datasets, and prompting formats (`zero-shot`, `cot`, `multi-turn`).
2. **Structured Reasoning (`cot-json`)**: Evaluating structured JSON schema extraction with explicit reasoning chains.
3. **Quantization Impact (`quantization`)**: Comparing floating point (`bf16`) vs. 4-bit NormalFloat (`nf4`) representations.
4. **Context & Adaptation (`fine-tuning`)**: Parameter-efficient adaptation (PEFT / LoRA) and multi-turn context length scaling.

---

## 🚀 Quick Start

### 1. Environment Setup

Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/audio-llm-ser-evaluation.git
cd audio-llm-ser-evaluation
pip install -r requirements.txt
```

### 2. Data Acquisition & Preprocessing

Download and extract datasets using provided acquisition modules:
```bash
# Acquisition
bash src/data_acquisition/meld/meld.sh
bash src/data_acquisition/iemocap/iemocap.sh

# Audio normalization (16kHz mono) and split preparation
python -m src.preprocessing.preprocess_meld
python -m src.preprocessing.preprocess_iemocap
```

### 3. Running Experiments

Execute experiment matrices directly using the inference runner:

```bash
# Dry run: preview resolved experiment matrix without execution
python -m src.inference.runner --dry-run

# Run the full main benchmark suite
python -m src.inference.runner --group main

# Run quantization ablations on MELD only
python -m src.inference.runner --group quantization --filter-dataset meld
```

### 4. Evaluation & Results

Evaluation outputs and predictions are organized systematically in the `results/` directory:
- Run metadata & resolved configuration snapshots (`resolved_config.json`)
- Per-sample predictions & reasoning logs (`predictions.csv`)
- Aggregated metrics: Macro-F1, Weighted-F1, Per-Class F1, Confusion Matrices

---

## 📈 Research Questions Explored

This repository is built to systematically answer key empirical questions in affective speech processing:
- **Zero-Shot Capability:** What emotional representations are natively captured by current LALMs?
- **Acoustic vs. Text Bias:** Does the model prioritize acoustic prosody over transcripts, or vice-versa?
- **Chain-of-Thought Utility:** Does explicit reasoning improve emotional nuance grounding and edge-case classification?
- **Robustness:** How do quantization (NF4 vs BF16), background noise, and unseen speakers affect classification consistency?
- **Domain Transfer:** How well do models generalize across heterogeneous emotion datasets (MELD $\leftrightarrow$ IEMOCAP)?
