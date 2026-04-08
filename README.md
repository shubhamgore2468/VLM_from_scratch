# VLM - Vision Language Model Training

A modular PyTorch implementation for training vision-language models with:
- **SigLIP** vision encoder (frozen)
- **Qwen2.5-0.5B** text decoder (frozen)
- **MLP projector** (trainable)
- **KV-cache** optimized inference

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/VLM.git
cd VLM

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Quick Start

### Training

```python
from VLM import TrainingConfig, train

# Configure training
config = TrainingConfig(
    data_path="/kaggle/working/vqa_llava_train.json",
    output_dir="/kaggle/working/checkpoints",
    epochs=1,
    micro_batch_size=4,
    grad_accum_steps=8,
)

# Start training
train(config)
```

### Inference

```python
from VLM import VLMInference
from PIL import Image

# Load model with trained projector
vlm = VLMInference(
    projector_checkpoint="/kaggle/input/models/foxtrot22/customvlm/pytorch/default/1/final.pt",
    device="cuda",
)

# Generate answer
image = Image.open("test.jpg").convert("RGB")
answer = vlm.generate(image, "What is in this image?", max_new_tokens=50)
print(answer)
```

### Prepare Dataset

```python
from VLM.data import prepare_vqa_samples, save_vqa_dataset

# Load samples from LLaVA-Instruct-150K
samples = prepare_vqa_samples(max_samples=20000)

# Save to JSON
save_vqa_dataset(samples, output="/kaggle/working/", dataset="llava", split="train")
```

## Project Structure

```
VLM/
├── __init__.py           # Package exports
├── data/
│   ├── __init__.py
│   └── dataset.py        # VQADataset, data utilities
├── engine/
│   ├── __init__.py
│   └── utils.py          # Checkpointing, distributed utils
├── inference/
│   ├── __init__.py
│   └── vlm_inference.py  # KV-cache inference
├── models/
│   ├── __init__.py
│   ├── projector.py      # MLP projector
│   └── vlm.py            # VLMTraining wrapper
├── training/
│   ├── __init__.py
│   ├── config.py         # TrainingConfig dataclass
│   └── train.py          # Main training loop
├── requirements.txt
├── setup.py
└── README.md
```

## Features

- **Efficient Training**: Gradient accumulation, AMP, gradient checkpointing
- **Multi-GPU**: DataParallel support out of the box
- **KV-Cache Inference**: ~2-3x speedup for generation
- **Modular Design**: Easy to swap vision/text models
- **WandB Logging**: Optional experiment tracking

## Kaggle Usage

```python
# In a Kaggle notebook:
!git clone https://github.com/yourusername/VLM.git
!pip install -e VLM/

from VLM import TrainingConfig, train, VLMInference

# Prepare data
from VLM.data import prepare_vqa_samples, save_vqa_dataset
samples = prepare_vqa_samples(max_samples=20000)
save_vqa_dataset(samples, output="/kaggle/working/")

# Train
config = TrainingConfig()
train(config)

# Inference
vlm = VLMInference(projector_checkpoint="/kaggle/working/checkpoints/final.pt")
```

## Benchmarks

KV-cache speedup (Kaggle T4 GPU):

| Tokens | Naive (s) | KV-Cache (s) | Speedup |
|--------|-----------|--------------|---------|
| 50     | 1.2       | 0.5          | 2.4x    |
| 100    | 2.8       | 0.9          | 3.1x    |
| 200    | 6.5       | 1.7          | 3.8x    |
