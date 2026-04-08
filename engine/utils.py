"""Training utilities: checkpointing, distributed training, etc."""

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler


def setup_distributed() -> Tuple[int, int, int]:
    """
    Initialize distributed training.
    
    Returns:
        Tuple of (rank, world_size, local_rank)
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    epoch: int,
    step: int,
    loss: float,
    config,
    path: str,
):
    """
    Save training checkpoint.
    
    Only saves projector weights (vision/text models are frozen).
    
    Args:
        model: VLM model (may be wrapped in DataParallel/DDP)
        optimizer: Optimizer state
        scheduler: LR scheduler state
        scaler: GradScaler state (for AMP)
        epoch: Current epoch
        step: Global step count
        loss: Current loss value
        config: Training config
        path: Save path
    """
    # Handle DataParallel/DDP wrapper
    if hasattr(model, "module"):
        projector_state = model.module.projector.state_dict()
    else:
        projector_state = model.projector.state_dict()
    
    checkpoint = {
        "projector_state_dict": projector_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "config": config.__dict__,
    }
    
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        path: Checkpoint path
        model: Model to load weights into
        optimizer: Optional optimizer to restore
        scheduler: Optional scheduler to restore
        scaler: Optional GradScaler to restore
        
    Returns:
        Checkpoint dict with metadata
    """
    checkpoint = torch.load(path, map_location="cpu")
    
    # Load projector weights
    if isinstance(model, DDP) or hasattr(model, "module"):
        model.module.projector.load_state_dict(checkpoint["projector_state_dict"])
    else:
        model.projector.load_state_dict(checkpoint["projector_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    return checkpoint