# Generative Adversarial Unlearning (GAU) for LLMs

This repository implements **Generative Adversarial Unlearning (GAU)** for Large Language Models (LLMs), enabling the removal of specific knowledge (unlearning) while retaining general capabilities. The pipeline is designed for research on machine unlearning, adversarial training, and evaluation of LLMs.

---

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Data Preparation](#data-preparation)
- [Running the Pipeline](#running-the-pipeline)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Features
- **Adversarial Unlearning:** Implements a GAN-like setup with a generator (unlearned model) and discriminator (evaluation model).
- **Flexible Data Pipeline:** Supports per-source and combined-source experiments, with robust Q-R (query-response) data formatting.
- **Offline Model Loading:** Supports loading large models (e.g., Llama-3-8B) from local disk for offline use.
- **Resource Monitoring:** Logs GPU/CPU/memory usage and supports CUDA profiling.
- **Mixed Precision & Gradient Accumulation:** Optimized for large models and limited GPU memory.
- **Extensive Configurability:** All major parameters are set via YAML config files.

---

## Project Structure
```
.
├── configs/                # YAML config files for model, training, data, evaluation
├── data/                   # Raw and processed data, including Q-R pairs
├── gau/                    # Core Python modules (models, training, evaluation, utils)
├── jobs/                   # SLURM job scripts and logs
├── pretrained/             # Local Hugging Face model directories
├── scripts/                # Data processing and Q-R generation scripts
├── main.py                 # Main training and evaluation pipeline
├── README.md               # This file
└── requirements.txt        # Python dependencies (if used)
```

---

## Setup & Installation

### 1. **Clone the repository**
```bash
git clone <your-repo-url>
cd <your-repo>
```

### 2. **Set up a Conda environment (Python 3.10 recommended)**
```bash
conda create -n gau-cuda python=3.10
conda activate gau-cuda
```

### 3. **Install PyTorch with CUDA (for CUDA 11.8)**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 4. **Install other dependencies**
```bash
pip install transformers pandas pyarrow tqdm pyyaml
```

### 5. **Download and place pretrained models**
- Download the required model (e.g., Llama-3-8B-Instruct) from Hugging Face and place it under `pretrained/Meta-Llama-3-8B-Instruct/`.

---

## Data Preparation

### 1. **Raw Data**
- Place your raw data (e.g., RedPajama, ArXiv, Wikipedia, etc.) in the appropriate subfolders under `data/`.

### 2. **Convert Raw Data to Q-R Format**
- Use the provided script to convert, for example, ArXiv parquet files to Q-R JSONL:
  ```bash
  python scripts/arxiv_to_qr.py
  ```
- This will create:
  - `data/processed/arxiv/target_qr.jsonl`
  - `data/processed/arxiv/retained_qr.jsonl`

### 3. **Update Configs**
- Edit `configs/data_config.yaml` to point to your processed Q-R files:
  ```yaml
  target_qr_file: data/processed/arxiv/target_qr.jsonl
  retained_qr_file: data/processed/arxiv/retained_qr.jsonl
  ```

---

## Running the Pipeline

### 1. **Edit Model and Training Configs**
- Set model and tokenizer paths in `configs/model_config.yaml` (use local paths for offline use).
- Adjust batch size, gradient accumulation, and other parameters in `configs/training_config.yaml` as needed for your GPU.

### 2. **Run the Main Pipeline**
```bash
python main.py
```
or submit a SLURM job:
```bash
sbatch jobs/run-main.sh
```

### 3. **Monitor Outputs**
- Logs and checkpoints will be saved as specified in your configs and job scripts.

---

## Configuration

- **configs/model_config.yaml:** Model and tokenizer paths, device, sequence length, etc.
- **configs/training_config.yaml:** Batch size, learning rate, optimizer, gradient accumulation, etc.
- **configs/data_config.yaml:** Paths to Q-R files, number of samples, etc.
- **configs/eval_config.yaml:** Evaluation metrics, thresholds, etc.

---

## Troubleshooting

- **CUDA Out of Memory:**  
  Reduce `batch_size`, increase `gradient_accumulation_steps`, or use a smaller model for the discriminator.
- **Tokenizer Padding Error:**  
  The code sets `tokenizer.pad_token = tokenizer.eos_token` if not already set.
- **DataLoader Collate Error:**  
  The pipeline uses a robust, tokenizer-aware collate function to ensure consistent batching.
- **Only 32GB GPU Memory Detected:**  
  You are likely on a V100-32GB GPU, not an A100-80GB. Check with `nvidia-smi`.

---

## Citation

If you use this codebase in your research, please cite the original GAU paper and this repository.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [RedPajama Dataset](https://github.com/togethercomputer/RedPajama-Data)
