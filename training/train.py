"""Main training loop for VLM."""

import os
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from transformers import (
    AutoProcessor,
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

from .config import TrainingConfig
from ..data.dataset import VQADataset, collate_fn
from ..data.ray_dataloader import RayDataloader
from ..models.projector import create_projector
from ..models.vlm import VLMTraining
from ..engine.utils import save_checkpoint, load_checkpoint


def train(config: TrainingConfig, resume_path: Optional[str] = None):
    """
    Main training function with DataParallel support.
    
    Args:
        config: Training configuration
        resume_path: Optional path to checkpoint to resume from
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize wandb (optional)
    try:
        import wandb
        wandb.init(
            project="vlm-training",
            name=f"siglip-qwen-{config.micro_batch_size}x{config.grad_accum_steps}",
            config=config.__dict__,
        )
        use_wandb = True
    except ImportError:
        print("wandb not installed, skipping logging")
        use_wandb = False
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"Training with {num_gpus} GPU(s)")
    print(f"Effective batch size: {config.effective_batch_size}")
    
    # Load vision encoder
    print("Loading vision encoder...")
    vision_processor = AutoProcessor.from_pretrained(config.vision_model_name)
    vision_model = AutoModel.from_pretrained(
        config.vision_model_name,
        torch_dtype=config.dtype,
    ).to(device)
    vision_model.eval()
    
    # Load text decoder
    print("Loading text decoder...")
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use flash attention if available
    attn_impl = "flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager"
    text_model = AutoModelForCausalLM.from_pretrained(
        config.text_model_name,
        torch_dtype=config.dtype,
        attn_implementation=attn_impl,
    ).to(device)
    text_model.eval()
    
    # Get dimensions for projector
    vision_dim = vision_model.config.vision_config.hidden_size
    text_dim = text_model.config.hidden_size
    print(f"Vision dim: {vision_dim}, Text dim: {text_dim}")
    
    # Create projector
    projector = create_projector(
        vision_dim, text_dim,
        projector_type=config.projector_type,
        dtype=config.dtype,
    ).to(device)
    print(f"Projector params: {sum(p.numel() for p in projector.parameters()):,}")
    
    # Wrap in training model
    model = VLMTraining(vision_model, text_model, projector, config)
    
    # DataParallel for multi-GPU
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    #Load dataset with RayDataloader or PyTorch DataLoader
    if config.use_ray_dataloader:
        dataloader = RayDataloader(
            data_path=config.data_path,
            batch_size=config.micro_batch_size,
            num_workers=config.num_workers,
        )
        dataset_size = 90672  # Or read from JSON
    else:
        dataset = VQADataset(
            config.data_path,
            vision_processor,
            tokenizer,
            config,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.micro_batch_size,
            shuffle=True,
            num_workers=config.num_workers, # parallel CPU workers for loading data
            pin_memory=True, # faster transfer CPU->GPU
            prefetch_factor=config.prefetch_factor,
            collate_fn=collate_fn, # custom batch fun - text length s vary
            drop_last=True, # drops incomplete batch
        )
        dataset_size = len(dataset)
    

    # Optimizer (only projector params)
    if hasattr(model, "module"):
        optimizer_params = model.module.projector.parameters()
    else:
        optimizer_params = model.projector.parameters()
    
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # LR scheduler
    total_steps = dataset_size // config.effective_batch_size * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # AMP scaler
    scaler = GradScaler("cuda") if config.use_amp and device.type == "cuda" else None
    
    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    
    if resume_path:
        print(f"Resuming from {resume_path}")
        ckpt = load_checkpoint(resume_path, model, optimizer, scheduler, scaler)
        start_epoch = ckpt["epoch"]
        global_step = ckpt["step"]
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    print("=" * 50)
    print("Starting training...")
    print("=" * 50)

    #Calc early stop step:
    config.early_stop_step = dataset_size // config.effective_batch_size

    
    for epoch in range(start_epoch, config.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        
        for step, batch in enumerate(pbar):
            if config.use_ray_dataloader:
                # Decode image bytes to tensor
                from PIL import Image
                import io
                images = [Image.open(io.BytesIO(b)).convert("RGB") for b in batch["image_bytes"]]
                pixel_values = vision_processor(images=images, return_tensors="pt")["pixel_values"]
                pixel_values = pixel_values.to(device, dtype=config.dtype)
            else:
                pixel_values = batch["pixel_values"].to(device, dtype=config.dtype)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prompt_lens = batch["prompt_lens"] if config.use_ray_dataloader else batch["prompt_len"]

            
            # Forward pass with AMP
            if config.use_amp:
                with autocast(device_type=device.type, dtype=config.dtype):
                    loss = model(pixel_values, input_ids, attention_mask, prompt_lens)
                    if loss.dim() > 0:  # DataParallel returns vector
                        loss = loss.mean()
                    loss = loss / config.grad_accum_steps
                scaler.scale(loss).backward()
            else:
                loss = model(pixel_values, input_ids, attention_mask, prompt_lens)
                if loss.dim() > 0:
                    loss = loss.mean()
                loss = loss / config.grad_accum_steps
                loss.backward()
            
            epoch_loss += loss.item() * config.grad_accum_steps
            num_batches += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item() * config.grad_accum_steps:.4f}",
                "gpu_mem": f"{torch.cuda.memory_allocated() / 1e9:.1f}GB",
            })
            
            # Gradient accumulation step
            if (step + 1) % config.grad_accum_steps == 0:
                if config.use_amp:
                    scaler.unscale_(optimizer)
                
                # Gradient clipping
                if hasattr(model, "module"):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.module.projector.parameters(),
                        config.max_grad_norm,
                    )
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.projector.parameters(),
                        config.max_grad_norm,
                    )
                
                if config.use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1

                # Early stopping
                if config.early_stop_step and global_step >= config.early_stop_step:
                    print(f"Early stopping at step {global_step}")
                    save_checkpoint(
                        model, optimizer, scheduler, scaler,
                        epoch, global_step, epoch_loss / num_batches,
                        config,
                        os.path.join(config.output_dir, f"early_stop_{global_step}.pt"),
                    )
                    if use_wandb:
                        wandb.finish()
                    return
                
                # Logging
                if global_step % config.log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = scheduler.get_last_lr()[0]
                    gpu_mem = torch.cuda.memory_allocated() / 1e9
                    
                    print(
                        f"\nStep {global_step} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Grad Norm: {grad_norm:.2f} | "
                        f"GPU Mem: {gpu_mem:.1f}GB"
                    )
                    
                    if use_wandb:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/lr": lr,
                            "train/grad_norm": grad_norm,
                            "train/step": global_step,
                            "system/gpu_memory_gb": gpu_mem,
                        }, step=global_step)
                
                # Save checkpoint
                if global_step % config.save_interval == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, scaler,
                        epoch, global_step, epoch_loss / num_batches,
                        config,
                        os.path.join(config.output_dir, f"step_{global_step}.pt"),
                    )
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1} complete | Avg Loss: {avg_epoch_loss:.4f}")
        print(f"{'=' * 50}\n")
        
        if use_wandb:
            wandb.log({"epoch/loss": avg_epoch_loss, "epoch/num": epoch + 1}, step=global_step)
        
        # Save epoch checkpoint
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            epoch + 1, global_step, avg_epoch_loss,
            config,
            os.path.join(config.output_dir, f"epoch_{epoch + 1}.pt"),
        )
    
    # Save final checkpoint
    save_checkpoint(
        model, optimizer, scheduler, scaler,
        config.epochs, global_step, avg_epoch_loss,
        config,
        os.path.join(config.output_dir, "final.pt"),
    )
    
    print("Training complete!")
    
    if use_wandb:
        wandb.finish()