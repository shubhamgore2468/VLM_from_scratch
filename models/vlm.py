"""VLM training model that combines vision encoder, projector, and text decoder."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..training.config import TrainingConfig


class VLMTraining(nn.Module):
    """
    Vision-Language Model wrapper for training.
    
    Combines a frozen vision encoder, trainable projector, and frozen text decoder.
    Only the projector is trained.
    """
    
    def __init__(
        self,
        vision_model: nn.Module,
        text_model: nn.Module,
        projector: nn.Module,
        config: TrainingConfig,
    ):
        super().__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        self.projector = projector
        self.config = config
        
        # Freeze vision and text models - only train projector
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        # Enable gradient checkpointing on LLM for memory efficiency
        if config.gradient_checkpointing and hasattr(self.text_model, "gradient_checkpointing_enable"):
            self.text_model.gradient_checkpointing_enable()
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lens: List[int],
    ) -> torch.Tensor:
        """
        Forward pass computing language modeling loss on answer tokens only.
        
        Args:
            pixel_values: Image tensors of shape (batch, 3, H, W)
            input_ids: Text token ids of shape (batch, seq_len)
            attention_mask: Attention mask of shape (batch, seq_len)
            prompt_lens: List of prompt lengths (to mask out question tokens)
            
        Returns:
            Cross-entropy loss scalar
        """
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
        # Extract vision features (frozen)
        with torch.no_grad():
            vision_out = self.vision_model.vision_model(pixel_values=pixel_values)
            vision_hidden = vision_out.last_hidden_state
        
        # Project to text space (trainable)
        projected = self.projector(vision_hidden)
        num_image_tokens = projected.shape[1]
        
        # Get text embeddings (frozen)
        with torch.no_grad():
            text_embeds = self.text_model.get_input_embeddings()(input_ids)
        
        # Concatenate [image_tokens, text_tokens]
        combined_embeds = torch.cat([projected, text_embeds], dim=1)
        
        # Build attention mask for combined sequence
        image_mask = torch.ones(
            batch_size,
            num_image_tokens,
            device=device,
            dtype=attention_mask.dtype,
        )
        combined_mask = torch.cat([image_mask, attention_mask], dim=1)
        
        # Forward through text model
        outputs = self.text_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            use_cache=False,
        )
        
        # Create labels: -100 for image tokens and prompt, actual ids for answer
        labels = torch.full(
            (batch_size, combined_embeds.shape[1]),
            -100,
            device=device,
            dtype=torch.long,
        )
        
        seq_len = input_ids.shape[1]
        
        for i in range(batch_size):
            # Answer starts after image tokens + prompt
            answer_start = num_image_tokens + prompt_lens[i]
            answer_end = num_image_tokens + seq_len
            labels[i, answer_start:answer_end] = input_ids[i, prompt_lens[i]:seq_len]
        
        # Compute cross-entropy loss
        logits = outputs.logits.float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        return loss