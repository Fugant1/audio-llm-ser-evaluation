# audio-llm-ser-evaluation

**Project Summary**
- **Goal:** Fine-tune an audio+text emotion model using MELD and IEMOCAP, applying LoRA (low-rank adaptation) for lightweight parameter-efficient fine-tuning.
- **Repo layout:** data download scripts in `src/data/*`, preprocessing in `preprocessing/`, model code in `models/`, and outputs in `results/`.

**Datasets**
- **MELD:** download and extract with the helper script [src/data/meld/meld.sh](src/data/meld/meld.sh#L1). A Python wrapper is available at [src/data/meld/get_meld_data.py](src/data/meld/get_meld_data.py#L1).
- **IEMOCAP:** see `src/data/iemocap/get_iemocap_data.py` for the acquisition script.

**Preprocessing**
- **Script:** run dataset-specific preprocessing in `preprocessing/` (e.g. [preprocessing/preprocess_meld.py](preprocessing/preprocess_meld.py#L1)). Preprocessing converts raw audio and transcripts into model-ready features and train/dev/test splits.

**Experiment / LoRA fine-tuning**
- **Approach:** use LoRA (low-rank adapters) to fine-tune a base transformer/multimodal model while keeping the base weights frozen. This reduces GPU memory and speeds up iteration.
- **Typical hyperparameters:** `rank=8`, `alpha=16`, `dropout=0.1`, learning rate `1e-4`–`3e-5`, batch size and epochs tuned per dataset. Use mixed precision (FP16) and gradient accumulation when needed.
- **Example (conceptual) command:**
	- `python train.py --dataset meld --model MODEL_NAME --lora_rank 8 --lora_alpha 16 --lora_dropout 0.1 --lr 3e-5 --batch_size 16 --epochs 5 --fp16`
	- Replace `train.py` and `MODEL_NAME` with your training entrypoint and chosen base model.

**Evaluation & Results**
- Results are written to `results/meld/` and `results/iemocap/`. Track: validation loss, accuracy/F1 per emotion class, confusion matrices, and checkpointed LoRA weights.

**Safety & reproducibility notes**
- The MELD download site may present TLS certificates signed by an InCommon CA. If `wget` fails with a certificate verification error, prefer adding the missing CA to your system trust store rather than using `--no-check-certificate` (which is vulnerable to MITM attacks). See `src/data/meld/meld.sh` and `src/data/meld/get_meld_data.py` for local-test alternatives.
- To test the download/extraction flow without downloading GBs, create a small fake `MELD.Raw.tar.gz` (see `src/data/meld/meld.sh` tests used during development) and run the script locally.

**Quick start**
- Download datasets: `bash src/data/meld/meld.sh` and the IEMOCAP script.
- Preprocess: `python preprocessing/preprocess_meld.py` (adjust filenames as needed).
- Train with LoRA: run your training script with LoRA flags (example above).
