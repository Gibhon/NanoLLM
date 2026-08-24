from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Config:
    # ---------------- Model ----------------
    embed_dimension: int = 128
    n_attention_heads: int = 4
    n_layers: int = 3
    expansion_factor: int = 4
    n_feedforward: int = embed_dimension * expansion_factor
    block_size: int = 128

    # ---------------- Training ----------------
    batch_size: int = 128
    n_workers: int = 4
    epochs: int = 5
    learning_rate: float = 5e-4
    weight_decay: float = 0.05

    # Generation
    max_new_tokens: int = 32
    sampling_temperature: float = 0.7
    top_k_candidate_count: int | None = None
    max_context_size: int = block_size

    # ---------------- Dropout ----------------
    dropout_pytorch: float = 0.2
    dropout_scratch: float = 0.3

    # ---------------- Paths ----------------
    data_path: Path = Path(__file__).resolve().parent / "data"
    model_path: Path = Path(__file__).resolve().parent / "checkpoints"

    # ---------------- Device ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
