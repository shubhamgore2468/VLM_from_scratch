"""Vision-to-text projector modules."""

import torch
import torch.nn as nn


class LinearProjector(nn.Module):
    """
    MLP projector that maps vision encoder outputs to text embedding space.
    
    Architecture: LayerNorm -> Linear -> GELU -> Linear
    """
    
    def __init__(self, vision_dim: int, text_dim: int, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.norm = nn.LayerNorm(vision_dim)
        self.fc1 = nn.Linear(vision_dim, text_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(text_dim, text_dim)
        
        # Convert to specified dtype
        self.to(dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project vision features to text embedding space.
        
        Args:
            x: Vision features of shape (batch, num_patches, vision_dim)
            
        Returns:
            Projected features of shape (batch, num_patches, text_dim)
        """
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def create_projector(
    vision_dim: int,
    text_dim: int,
    projector_type: str = "mlp",
    dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """
    Factory function to create projector modules.
    
    Args:
        vision_dim: Dimension of vision encoder outputs
        text_dim: Dimension of text model embeddings
        projector_type: Type of projector ("mlp" supported)
        dtype: Data type for projector weights
        
    Returns:
        Projector module
    """
    if projector_type == "mlp":
        return LinearProjector(vision_dim, text_dim, dtype)
    else:
        raise ValueError(f"Unknown projector type: {projector_type}")