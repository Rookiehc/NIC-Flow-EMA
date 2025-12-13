import math
import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, embedding_type: str = "positional", use_mlp: bool = True):
        super().__init__()
        self.use_mlp = use_mlp
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.embedding_type = embedding_type
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(frequency_embedding_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )

    @staticmethod
    def positional_timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float64, device=t.device) / half)
        args = t[:, None].to(torch.float64) * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if self.embedding_type == "positional":
            t_freq = self.positional_timestep_embedding(t, self.frequency_embedding_size)
        else:
            # default positional
            t_freq = self.positional_timestep_embedding(t, self.frequency_embedding_size)
        if self.use_mlp:
            t_emb = self.mlp(t_freq.to(dtype=t.dtype))
        else:
            t_emb = t_freq.to(dtype=t.dtype)
        return t_emb


class TimeFusion(nn.Module):
    """Fuse primary time t and secondary time s into a single embedding vector.
    Intended to be added to UNet time embedding path via external hook.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.t_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size)
        self.s_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size)

    def forward(self, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_embedder(t)
        s_emb = self.s_embedder(s)
        return t_emb + s_emb
