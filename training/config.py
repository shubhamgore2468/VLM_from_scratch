"""Training configuration dataclass."""

from dataclasses import dataclass
import torch
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for VLM training."""
    
    # Model settings
    vision_model_name: str = "google/siglip-base-patch16-224"
    text_model_name: str = "Qwen/Qwen2.5-0.5B"
    image_size: int = 224
    patch_size: int = 16
    max_text_length: int = 256
    
    # Training hyperparameters
    epochs: int = 1
    micro_batch_size: int = 4 # Effective batch size = 4*8*2=64 = micro_batch_size * grad_accum_steps * num_gpus
    grad_accum_steps: int = 8
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Precision and optimization
    dtype: torch.dtype = torch.float16
    use_amp: bool = True
    use_compile: bool = False
    gradient_checkpointing: bool = True
    early_stop_step: Optional[int] = None
    
    # Data loading
    num_workers: int = 4
    prefetch_factor: int = 2
    data_path: str = "/kaggle/working/vqa_llava_train.json"
    use_ray_dataloader: bool = True
    
    # Checkpointing and logging
    output_dir: str = "./checkpoints"
    log_interval: int = 10
    save_interval: int = 500
    
    # Projector architecture
    projector_type: str = "mlp"
    projector_depth: int = 2
    
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size including gradient accumulation."""
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        return self.micro_batch_size * self.grad_accum_steps * num_gpus