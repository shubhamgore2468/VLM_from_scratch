import ray
from ray.data import Dataset
from typing import Dict, Any
import torch
from .ray_pipeline import PipelineConfig, create_pipeline


class RayDataloader:
    '''
    RayDataloader wraps a Ray Dataset to provide an iterable dataloader interface.
    '''
    def __init__(
        self,
        data_path: str,
        batch_size: int = 4,
        image_base_dir: str = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017",
        #/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017
        num_workers: int = 4,
        prefetch_batches: int = 2,
        vision_model_name: str = "google/siglip-base-patch16-224",
        text_model_name: str = "Qwen/Qwen2.5-0.5B",
        device: str = "cuda",
    ):
        self.config = PipelineConfig(
            input_path=data_path,
            output_path="",  # Not saving
            image_base_dir=image_base_dir,
            num_workers=num_workers,
            batch_size=batch_size * prefetch_batches,
            vision_model_name=vision_model_name,
            text_model_name=text_model_name,
        )

        self.batch_size = batch_size
        self.device = device
        self._ds = None

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    def _get_dataset(self):
        if self._ds is None:
            self._ds = create_pipeline(self.config)
        return self._ds
    
    def __iter__(self):
        ds = self._get_dataset()
        for batch in ds.iter_torch_batches(
            batch_size = self.batch_size
        ):
            yield{ # yield doesn;t return moves to next batch after this
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }


    def __len__(self):
        return self._get_dataset().count() // self.batch_size
