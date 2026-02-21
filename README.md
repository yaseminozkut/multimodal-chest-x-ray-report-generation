# Chest X-Ray Report Generation with LLaMA 3.2 Vision

Fine-tuned LLaMA 3.2 Vision (11B) using 4-bit QLoRA for automated radiology report generation from chest X-ray images.

## Highlights

- **Model**: LLaMA 3.2-11B-Vision-Instruct with LoRA adapters (r=16, alpha=16)
- **Quantization**: 4-bit QLoRA via [Unsloth](https://github.com/unslothai/unsloth) for memory-efficient training
- **Dataset**: Curated 100K+ MIMIC-CXR chest X-ray images (multi-view with matched reports)
- **Training**: Hugging Face TRL SFTTrainer on HPC with SLURM
- **Early Stopping**: Patience=3 to prevent overfitting

## Results

| Metric | Score |
|--------|-------|
| BLEU-1 | 28.12 |
| BLEU-4 | 7.45 |
| ROUGE-1 | 39.45 |
| ROUGE-L | 26.70 |
| Training Loss | 2.26 → 0.48 |

### Sample Outputs

**Example 1:**
> Lungs are fully expanded and clear. No pleural abnormalities. Heart size is normal. Cardiomediastinal and hilar silhouettes are unremarkable.

**Example 2:**
> The heart is not enlarged. Mediastinal and hilar contours are within normal limits. Pulmonary vascularity is normal. No pleural effusion, focal consolidation, or pneumothorax is seen.

**Example 3:**
> The lungs are hyperinflated with severe bullous emphysema. There is new mild enlargement of the cardiac silhouette. The mediastinal and hilar contours are unchanged.

## Project Structure

```
├── train.py                 # Main training script
├── train_single_gpu.py      # Single GPU training (alternative)
├── train.slurm              # SLURM job submission script
├── dataset.py               # Data loader
├── xray_dataset.py          # X-ray dataset class
├── similarity_metrics.py    # BLEU, ROUGE evaluation
├── calculate_metrics.py     # Quick metrics calculation
├── check_report.py          # Report utilities
├── requirements.txt
├── cache/                   # Auto-generated dataset cache
└── outputs/                 # Model checkpoints (gitignored)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
# Copy the example env file and fill in your paths
cp .env.example .env

# Edit .env with your actual paths
DATASET_DIR=/path/to/your/cleaned_dataset
METADATA_FILE=/path/to/your/metadata.xlsx
```

### 3. Train

```bash
# Single GPU training
python train.py

# With SLURM (recommended)
sbatch train.slurm

# Or specify a GPU type
sbatch --gres=gpu:a6000:1 train.slurm
```

Note: First run converts the dataset (~17 min), subsequent runs load from cache.

## Dataset

This project uses [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/), which requires PhysioNet credentialed access. The dataset was curated to match PA/lateral views with corresponding Findings sections from radiology reports.

## Tech Stack

- [Unsloth](https://github.com/unslothai/unsloth) - Memory-efficient LoRA fine-tuning
- [Hugging Face TRL](https://github.com/huggingface/trl) - SFTTrainer
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning
- [SLURM](https://slurm.schedmd.com/) - HPC job scheduling

## License

MIT
