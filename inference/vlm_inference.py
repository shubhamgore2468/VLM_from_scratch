"""VLM inference with KV-cache optimization."""

from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    AutoProcessor,
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from PIL import Image


class VLMInference(nn.Module):
    """
    Inference module for VLM with KV-cache support.
    
    Supports both naive generation (for benchmarking) and 
    KV-cache optimized generation.
    """
    
    def __init__(
        self,
        vision_model_name: str = "google/siglip-base-patch16-224",
        text_model_name: str = "Qwen/Qwen2.5-0.5B",
        projector_checkpoint: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize VLM for inference.
        
        Args:
            vision_model_name: HuggingFace vision model name
            text_model_name: HuggingFace text model name
            projector_checkpoint: Path to trained projector checkpoint
            device: Device to run inference on
            dtype: Data type for model weights
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        # Load vision encoder
        print("Loading vision encoder...")
        self.vision_processor = AutoProcessor.from_pretrained(vision_model_name)
        self.vision_model = AutoModel.from_pretrained(
            vision_model_name,
            torch_dtype=dtype,
        ).to(device)
        self.vision_model.eval()
        
        # Load text decoder
        print("Loading text decoder...")
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.text_model = AutoModelForCausalLM.from_pretrained(
            text_model_name,
            torch_dtype=dtype,
        ).to(device)
        self.text_model.eval()
        
        # Get dimensions
        vision_dim = self.vision_model.config.vision_config.hidden_size
        text_dim = self.text_model.config.hidden_size
        
        # Create projector (same architecture as training)
        print("Creating projector...")
        self.projector = self._create_projector(vision_dim, text_dim).to(device)
        
        # Load trained weights if provided
        if projector_checkpoint:
            self._load_projector(projector_checkpoint)
        
        # Freeze everything for inference
        for param in self.parameters():
            param.requires_grad = False
        
        print(f"VLM ready! Vision: {vision_dim}, Text: {text_dim}")
    
    def _create_projector(self, vision_dim: int, text_dim: int) -> nn.Module:
        """Create projector matching training architecture."""
        return nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim),
        ).to(self.dtype)
    
    def _load_projector(self, checkpoint_path: str):
        """Load projector weights from checkpoint."""
        print(f"Loading projector from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state_dict = ckpt["projector_state_dict"]
        
        # Map from named module to sequential indices
        new_state_dict = {
            "0.weight": state_dict["norm.weight"],
            "0.bias": state_dict["norm.bias"],
            "1.weight": state_dict["fc1.weight"],
            "1.bias": state_dict["fc1.bias"],
            "3.weight": state_dict["fc2.weight"],
            "3.bias": state_dict["fc2.bias"],
        }
        
        self.projector.load_state_dict(new_state_dict)
        print(f"Loaded! (loss={ckpt.get('loss', 'N/A'):.4f}, step={ckpt.get('step', 'N/A')})")
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode image to projected embeddings.
        
        Args:
            image: PIL Image
            
        Returns:
            Projected embeddings of shape (1, num_patches, text_dim)
        """
        pixel_values = self.vision_processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"].to(self.device, dtype=self.dtype)
        
        with torch.no_grad():
            vision_out = self.vision_model.vision_model(pixel_values=pixel_values)
            vision_hidden = vision_out.last_hidden_state
        
        projected = self.projector(vision_hidden)
        return projected
    
    def prepare_prompt(self, question: str) -> dict:
        """Prepare tokenized prompt."""
        prompt = f"Question: {question} Answer:"
        tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        return {
            "input_ids": tokens["input_ids"].to(self.device),
            "attention_mask": tokens["attention_mask"].to(self.device),
        }
    
    @torch.no_grad()
    def generate_naive(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> str:
        """
        Generate WITHOUT KV-cache (slow baseline for benchmarking).
        
        Recomputes all attention for entire sequence at each step.
        """
        image_embeds = self.encode_image(image)
        prompt_data = self.prepare_prompt(question)
        input_ids = prompt_data["input_ids"]
        text_embeds = self.text_model.get_input_embeddings()(input_ids)
        
        combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        attention_mask = torch.ones(
            1, combined_embeds.shape[1], device=self.device, dtype=torch.long
        )
        
        generated_ids = []
        
        for step in range(max_new_tokens):
            outputs = self.text_model(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
            
            next_token_logits = outputs.logits[:, -1, :]
            
            if do_sample and temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            generated_ids.append(next_token.item())
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Append new token embedding to sequence
            next_token_embed = self.text_model.get_input_embeddings()(next_token)
            combined_embeds = torch.cat([combined_embeds, next_token_embed], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, 1, device=self.device, dtype=torch.long),
            ], dim=1)
        
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    @torch.no_grad()
    def generate_with_kv_cache(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> str:
        """
        Generate WITH KV-cache (optimized).
        
        Prefills cache with full sequence, then only processes
        one new token at each step.
        """
        image_embeds = self.encode_image(image)
        prompt_data = self.prepare_prompt(question)
        input_ids = prompt_data["input_ids"]
        text_embeds = self.text_model.get_input_embeddings()(input_ids)
        
        combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        attention_mask = torch.ones(
            1, combined_embeds.shape[1], device=self.device, dtype=torch.long
        )
        
        # PREFILL: Process all tokens, build cache
        outputs = self.text_model(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        
        next_token_logits = outputs.logits[:, -1, :]
        
        if do_sample and temperature > 0:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        
        generated_ids = [next_token.item()]
        
        if next_token.item() == self.tokenizer.eos_token_id:
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # DECODE: Process only new token, reuse cache
        for step in range(1, max_new_tokens):
            next_token_embed = self.text_model.get_input_embeddings()(next_token)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, 1, device=self.device, dtype=torch.long),
            ], dim=1)
            
            outputs = self.text_model(
                inputs_embeds=next_token_embed,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            
            next_token_logits = outputs.logits[:, -1, :]
            
            if do_sample and temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            generated_ids.append(next_token.item())
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    def generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        do_sample: bool = False,
        use_kv_cache: bool = True,
    ) -> str:
        """
        Generate answer for image+question.
        
        Args:
            image: PIL Image
            question: Question string
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (only if do_sample=True)
            do_sample: Whether to use sampling vs greedy decoding
            use_kv_cache: Whether to use KV-cache optimization
            
        Returns:
            Generated answer string
        """
        if use_kv_cache:
            return self.generate_with_kv_cache(
                image, question, max_new_tokens, temperature, do_sample
            )
        else:
            return self.generate_naive(
                image, question, max_new_tokens, temperature, do_sample
            )