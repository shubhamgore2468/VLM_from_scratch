# data/ray_dataloader.py - FIXED VERSION

import ray
from ray.data import Dataset
from typing import Dict, Any
import torch
from transformers import AutoProcessor
from PIL import Image
import io
from .ray_pipeline import PipelineConfig, create_pipeline
import logging


class RayDataloader:
    def __init__(
        self,
        data_path: str,
        batch_size: int = 4,
        image_base_dir: str = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017",
        num_workers: int = 2,
        prefetch_batches: int = 4,
        vision_model_name: str = "google/siglip-base-patch16-224",
        text_model_name: str = "Qwen/Qwen2.5-0.5B",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.config = PipelineConfig(
            input_path=data_path,
            output_path="",
            image_base_dir=image_base_dir,
            num_workers=num_workers,
            batch_size=batch_size * prefetch_batches,
            vision_model_name=vision_model_name,
            text_model_name=text_model_name,
        )

        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self._ds = None
        
        # Load vision processor for on-the-fly preprocessing
        self.vision_processor = AutoProcessor.from_pretrained(vision_model_name)

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=2, logging_level=logging.WARNING, log_to_driver=False)

    def _get_dataset(self):
        if self._ds is None:
            self._ds = create_pipeline(self.config)
        return self._ds
    
    def _process_images(self, image_bytes_list):
        """Convert raw image bytes to processed pixel_values tensor."""
        images = []
        for img_bytes in image_bytes_list:
            if img_bytes is not None:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            else:
                img = Image.new("RGB", (224, 224), (128, 128, 128))
            images.append(img)
        
        # Batch process with vision processor
        processed = self.vision_processor(images=images, return_tensors="pt")
        return processed["pixel_values"]
    
    def __iter__(self):
        ds = self._get_dataset()
        for batch in ds.iter_batches(batch_size=self.batch_size, batch_format="numpy"):
            # Process images to pixel_values
            pixel_values = self._process_images(batch["image_bytes"])
            
            yield {
                "pixel_values": pixel_values.to(self.device, dtype=self.dtype),
                "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long, device=self.device),
                "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long, device=self.device),
                "prompt_lens": batch["prompt_lens"].tolist(),
            }

    def __len__(self):
        return self._get_dataset().count() // self.batch_size