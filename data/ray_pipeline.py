from attr import dataclass
import ray
from ray.data import Dataset
import pyarrow as pa
from pathlib import Path
import json
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
import io
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModel

@dataclass
class PipelineConfig:
    input_path: str
    output_path: str
    image_base_dir: str = ""
    image_size: int = 224
    patch_size: int = 16
    max_text_length: int = 256
    num_workers: int = 4
    batch_size: int = 64
    vision_model_name: str = "google/siglip-base-patch16-224"
    text_model_name: str = "Qwen/Qwen2.5-0.5B"

def validate_sample(row, config: PipelineConfig):
    row["valid"] = bool(
        row.get("image") and 
        row.get("question") and 
        row.get("answer") and
        len(row["question"].strip()) > 0
    )
    return row


def load_image(row: Dict[str, Any], image_size: int = 224) -> Dict[str, Any]:
    """Load and preprocess image to bytes."""
    if not row.get("valid", True):
        row["image_bytes"] = None
        return row
    
    # Use path directly from JSON (supports absolute paths)
    img_path = Path(row["image"])
    
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        
        # Store as bytes for serialization
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        row["image_bytes"] = buffer.getvalue()
        row["valid"] = True
    except Exception as e:
        row["image_bytes"] = None
        row["valid"] = False
    
    return row

class TokenizeBatch:
    '''
    Tokenize text data in batches using pretrained tokenizer.
    '''
    def __init__(self, config: PipelineConfig):
        self.tokenizer  = AutoTokenizer.from_pretrained(config.text_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = config.max_text_length

    def __call__(self, batch):
        prompts = [f"Question: {q} Answer:" for q in batch["question"]]
        full_texts = [f"{p} {a}" for p, a in zip(prompts, batch["answer"])]
        
        prompt_lens = []
        for p in prompts:
            tokens = self.tokenizer(p, add_special_tokens=True, truncation=True, max_length=self.max_length)
            prompt_lens.append(len(tokens["input_ids"]))

        encoded = self.tokenizer( full_texts, add_special_tokens=True, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="np") # numpy as it is lightwight only for data processing step

        return {
            "image_bytes": batch["image_bytes"],
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "prompt_lens": np.array(prompt_lens),
        }
    
class ExtractVisionFeature:
    '''
    Extract vision features using pretrained model.
    '''

    def __init__(self, config: PipelineConfig):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vision_processor = AutoProcessor.from_pretrained(config.vision_model_name)
        self.model = AutoModel.from_pretrained(config.vision_model_name).to(self.device)
        self.model.eval()

    def __call__(self, batch):
        images = []
        for img_bytes in batch["image_bytes"]:
            if img_bytes is not None:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                images.append(img)
            else:
                images.append(Image.new("RGB", (224,224), (128, 128, 128)))

        inputs = self.processor(images=images, return_tensors='pt')
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            features = self.model.vision_model(pixel_values).last_hidden_state

        return {
            "vision_features": features.cpu().numpy(),
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "prompt_lens": batch["prompt_lens"],
        }
    

def create_pipeline(config: PipelineConfig):

    ds = ray.data.read_json(config.input_path) # ds is like a distributed table

    # validates the samples
    ds = ds.map(lambda row: validate_sample(row, config))    
    ds = ds.filter(lambda row: row["valid"])

    #load images
    ds = ds.map(lambda row: load_image(row, config.image_size), num_cpus=0.5) # each task uses half CPU core
    ds = ds.filter(lambda row: row["valid"])

    ds = ds.map_batches( # process data in batches
        TokenizeBatch,
        fn_constructor_kwargs = {"config": config},
        batch_size=config.batch_size,
        num_cpus=0.5,
    )
    return ds

def create_pipeline_with_features(config: PipelineConfig):
    ds = create_pipeline(config)
    ds = ds.map_batches(
        ExtractVisionFeature,
        fn_constructor_kwargs={"config": config}, # passes config to ToekzieBatch constructor
        batch_size=config.batch_size,
        num_cpus=0.5, # each task uses half CPU core
    )
    return ds

def save_to_parquet(ds: Dataset, output_path: str):
    """Save processed dataset to parquet."""
    ds.write_parquet(output_path)
    print(f"Saved to {output_path}")
 
 
def to_torch_dataloader(ds: Dataset, batch_size: int = 4):
    """Convert to PyTorch DataLoader for training."""
    return ds.iter_torch_batches(batch_size=batch_size, dtypes={"input_ids": "int64"})
 
def run_pipeline(config: PipelineConfig, extract_features: bool = False):
    """Run the full pipeline."""

    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    print(f"Processing {config.input_path}")
    print(f"Workers: {config.num_workers}")

    # Create pipeline
    if extract_features:
        ds = create_pipeline_with_features(config)
    else:
        ds = create_pipeline(config)

    # Save
    save_to_parquet(ds, config.output_path)

    # Stats
    print(f"Processed {ds.count()} samples")

    ray.shutdown()