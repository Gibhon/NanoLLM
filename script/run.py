import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config

SRC_PATH = Path(__file__).resolve().parent.parent / "src"

from engine import plot_model_comparison, train_and_val  # type: ignore

sys.path.append(str(SRC_PATH / "models"))
from pytorch_model import NanoLLM  # type: ignore
from scratch_model import NanoLLMScratch  # type: ignore

# -------Scratch Model-----------

sys.path.append(str(SRC_PATH / "data_manager"))
from data_cleaner import clean_data  # type: ignore
from dataset import NanoLLMDataset  # type: ignore
from tokenizer import CharTokenizer  # type: ignore

if __name__ == "__main__":
    # ---Config---
    cfg = Config()

    # ---Data---
    data = clean_data(cfg.data_path / "data.txt")

    tokenizer = CharTokenizer(data)
    token_ids = tokenizer.encode(data)

    split_idx = int(0.75 * len(token_ids))
    train_ids, val_ids = token_ids[:split_idx], token_ids[split_idx:]

    train_dataset = NanoLLMDataset(train_ids, cfg.block_size)
    val_dataset = NanoLLMDataset(val_ids, cfg.block_size)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.n_workers,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.n_workers,
    )

    # ---Model Pytorch---
    pytorch_model = NanoLLM(
        tokenizer,
        embed_dim=cfg.embed_dimension,
        n_heads=cfg.n_attention_heads,
        n_layers=cfg.n_layers,
        n_feedforward=cfg.n_feedforward,
        block_size=cfg.block_size,
        dropout=cfg.dropout_pytorch,
    ).to(cfg.device)

    # ---Model Scratch---
    scratch_model = NanoLLMScratch(
        tokenizer=tokenizer,
        max_seq_len=cfg.block_size,
        embed_dim=cfg.embed_dimension,
        n_heads=cfg.n_attention_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout_scratch,
    ).to(cfg.device)

    # ---Model Necessities---
    optimizer_pytorch = torch.optim.AdamW(
        pytorch_model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    optimizer_scratch = torch.optim.AdamW(
        scratch_model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scheduler_pytorch = CosineAnnealingLR(
        optimizer_pytorch,
        T_max=cfg.epochs,
    )

    scheduler_scratch = CosineAnnealingLR(
        optimizer_scratch,
        T_max=cfg.epochs,
    )

    loss_fn = F.cross_entropy

    # ---Train---
    pytorch_model_path = cfg.model_path / "pytorch_model.pth"
    scratch_model_path = cfg.model_path / "scratch_model.pth"

    models = {
        "pytorch": {
            "model": pytorch_model,
            "optimizer": optimizer_pytorch,
            "scheduler": scheduler_pytorch,
            "model_path": pytorch_model_path,
            "type": "Pytorch Model",
        },
        "scratch": {
            "model": scratch_model,
            "optimizer": optimizer_scratch,
            "scheduler": scheduler_scratch,
            "model_path": scratch_model_path,
            "type": "Scratch Model",
        },
    }

    results = {}

    for name, model_config in models.items():
        results[name] = train_and_val(
            model=model_config["model"],
            epochs=cfg.epochs,
            loss_fn=loss_fn,
            optimizer=model_config["optimizer"],
            train_loader=train_loader,
            val_loader=val_loader,
            device=cfg.device,
            model_path=model_config["model_path"],
            scheduler=model_config["scheduler"],
            type=model_config["type"],
        )

    loss_history_pytorch, min_train_loss_pytorch, min_val_loss_pytorch, time_pytorch = (
        results["pytorch"]
    )
    loss_history_scratch, min_train_loss_scratch, min_val_loss_scratch, time_scratch = (
        results["scratch"]
    )

    # ---Compare---
    plot_model_comparison(
        loss_history_pytorch=loss_history_pytorch,
        min_train_loss_pytorch=min_train_loss_pytorch,
        min_val_loss_pytorch=min_val_loss_pytorch,
        time_pytorch=time_pytorch,
        loss_history_scratch=loss_history_scratch,
        min_train_loss_scratch=min_train_loss_scratch,
        min_val_loss_scratch=min_val_loss_scratch,
        time_scratch=time_scratch,
        graph_path=cfg.data_path / "model_comparison7.png",
    )

    print("PYTORCH MODEL STATS:")
    print(f"Train Loss History : {loss_history_pytorch.get('train_loss')}")
    print(f"Val Loss History : {loss_history_pytorch.get('val_loss')}")
    print(f"Min Train Loss: {min_train_loss_pytorch}")
    print(f"Min Val Loss: {min_val_loss_pytorch}")

    print("SCRATCH MODEL STATS:")
    print(f"Train Loss History : {loss_history_scratch.get('train_loss')}")
    print(f"Val Loss History : {loss_history_scratch.get('val_loss')}")
    print(f"Min Train Loss: {min_train_loss_scratch}")
    print(f"Min Val Loss: {min_val_loss_scratch}")
