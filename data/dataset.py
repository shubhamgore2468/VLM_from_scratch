"""Dataset classes and data utilities for VLM training."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm


class VQADataset(Dataset):
    """
    Visual Question Answering dataset.
    
    Expected JSON format:
    [
        {"image": "path/to/image.jpg", "question": "Q", "answer": "A"},
        ...
    ]
    """
    
    def __init__(
        self,
        data_path: str,
        vision_processor,
        tokenizer,
        config,
        image_base_dir: Optional[str] = None,
    ):
        """
        Initialize VQA dataset.
        
        Args:
            data_path: Path to JSON file with samples
            vision_processor: HuggingFace vision processor
            tokenizer: HuggingFace tokenizer
            config: TrainingConfig instance
            image_base_dir: Base directory for image paths (default: data_path parent)
        """
        with open(data_path, "r") as f:
            self.samples = json.load(f)
        
        self.vision_processor = vision_processor
        self.tokenizer = tokenizer
        self.config = config
        self.image_base_dir = Path(image_base_dir) if image_base_dir else Path(data_path).parent
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        img_path = self.image_base_dir / sample["image"]
        
        # Load image with fallback for missing files
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            img = Image.new("RGB", (self.config.image_size, self.config.image_size), (128, 128, 128))
        
        # Process image
        pixel_values = self.vision_processor(
            images=img, return_tensors="pt"
        )["pixel_values"].squeeze(0)
        
        # Format prompt and full text
        prompt = f"Question: {sample['question']} Answer:"
        full_text = f"{prompt} {sample['answer']}"
        
        # Tokenize prompt to get length
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.config.max_text_length,
            add_special_tokens=True,
        )
        prompt_len = len(prompt_tokens["input_ids"])
        
        # Tokenize full text
        full_tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.config.max_text_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": full_tokens["input_ids"].squeeze(0),
            "attention_mask": full_tokens["attention_mask"].squeeze(0),
            "prompt_len": prompt_len,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for VQA batches."""
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "prompt_len": [x["prompt_len"] for x in batch],
    }


def prepare_vqa_samples(max_samples: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Load and prepare VQA samples from LLaVA-Instruct-150K.
    
    Args:
        max_samples: Maximum number of samples to load (None for all)
        
    Returns:
        List of sample dicts with image, question, answer keys
    """
    with open("/kaggle/input/datasets/foxtrot22/llava-instruct-150k/llava_instruct_150k.json") as f:
        data = json.load(f)

    # Process samples
    samples = []
    for item in data[:max_samples]:
        convs = item.get("conversations", [])
        if len(convs) < 2:
            continue
        
        for i in range(0, len(convs)-1, 2):
            if convs[i].get("from") == "human" and convs[i+1].get("from") == "gpt":
                question = convs[i].get("value", "").replace("<image>\n", "").strip()
                answer = convs[i+1].get("value", "").strip()
                
                if question and answer:
                    img_name = item.get("image", "")
                    samples.append({
                        "image": f"/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017/{img_name}",
                        "question": question,
                        "answer": answer
                    })

    return samples


def save_vqa_dataset(
    samples: List[Dict[str, str]],
    output: str = "./data",
    dataset: str = "llava",
    split: str = "train",
    seed: int = 42,
) -> Path:
    """
    Save VQA samples to JSON file.
    
    Args:
        samples: List of sample dicts
        output: Output directory
        dataset: Dataset name for filename
        split: Split name for filename
        seed: Random seed
        
    Returns:
        Path to saved file
    """
    random.seed(seed)
    
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"vqa_{dataset}_{split}.json"
    
    with open(output_file, "w") as f:
        json.dump(samples, f, indent=2)
    
    print(f"\n Saved {len(samples)} samples to {output_file}")
    print("\n Sample entry:")
    print(json.dumps(samples[0], indent=2))
    
    return output_file