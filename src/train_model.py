import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve()

from train import train_and_val, plot_model_comparison  # type: ignore

sys.path.append(str(SRC_PATH / "models"))
from pytorch_model import NanoLLM  # type: ignore
from scratch_model import NanoLLMScratch  # type: ignore

# -------Scratch Model-----------

sys.path.append(str(SRC_PATH / "data_manager"))
from data_cleaner import clean_data  # type: ignore
from tokenizer import CharTokenizer  # type: ignore
from dataset import NanoLLMDataset  # type: ignore

if __name__ == "__main__":
    # ---Config---
    EMBEDDING_DIMENSION = 128
    NUM_ATTENTION_HEADS = 4
    NUM_LAYERS = 3
    NUM_FEEDFORWARD = EMBEDDING_DIMENSION * 4
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    BLOCK_SIZE = 128

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 10
    LR = 5e-4
    DROPOUT_pytorch = 0.2
    DROPOUT_scratch = 0.3
    WEIGHT_DECAY = 0.05
    DATA_PATH = Path(__file__).resolve().parent.parent / "data"

    # ---Data---
    data = clean_data(DATA_PATH / "data.txt")
    tokenizer = CharTokenizer(data)
    token_ids = tokenizer.encode(data)

    split_idx = int(0.75 * len(token_ids))
    train_ids, val_ids = token_ids[:split_idx], token_ids[split_idx:]

    train_dataset = NanoLLMDataset(train_ids, BLOCK_SIZE)
    val_dataset = NanoLLMDataset(val_ids, BLOCK_SIZE)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # ---Model Pytorch---
    shakespeare_model_pytorch = NanoLLM(
        tokenizer,
        embed_dim=EMBEDDING_DIMENSION,
        n_heads=NUM_ATTENTION_HEADS,
        n_layers=NUM_LAYERS,
        n_feedforward=NUM_FEEDFORWARD,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT_pytorch,
    ).to(DEVICE)

    # ---Model Scratch---
    shakespeare_model_scratch = NanoLLMScratch(
        tokenizer=tokenizer,
        max_seq_len=BLOCK_SIZE,
        embed_dim=EMBEDDING_DIMENSION,
        n_heads=NUM_ATTENTION_HEADS,
        n_layers=NUM_LAYERS,
        dropout=DROPOUT_scratch,
    ).to(DEVICE)

    # ---Model Necessities---
    optimizer_pytorch = torch.optim.AdamW(
        shakespeare_model_pytorch.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    optimizer_scratch = torch.optim.AdamW(
        shakespeare_model_scratch.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler_pytorch = CosineAnnealingLR(optimizer_pytorch, T_max=EPOCHS)
    scheduler_scratch = CosineAnnealingLR(optimizer_scratch, T_max=EPOCHS)
    loss_fn = F.cross_entropy

    # ---Train---
    pytorch_model_path = DATA_PATH / "pytorch_model.pth"
    scratch_model_path = DATA_PATH / "scratch_model.pth"
    loss_history_pytorch, min_train_loss_pytorch, min_val_loss_pytorch, time_pytorch = (
        train_and_val(
            model=shakespeare_model_pytorch,
            epochs=EPOCHS,
            loss_fn=loss_fn,
            optimizer=optimizer_pytorch,
            train_loader=train_loader,
            val_loader=val_loader,
            device=DEVICE,
            model_path=pytorch_model_path,
            scheduler=scheduler_pytorch,
        )
    )
    loss_history_scratch, min_train_loss_scratch, min_val_loss_scratch, time_scratch = (
        train_and_val(
            model=shakespeare_model_scratch,
            epochs=EPOCHS,
            loss_fn=loss_fn,
            optimizer=optimizer_scratch,
            train_loader=train_loader,
            val_loader=val_loader,
            device=DEVICE,
            model_path=scratch_model_path,
            scheduler=scheduler_scratch,
        )
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
        graph_path=DATA_PATH / "model_comparison6.png",
    )

    print("PYTORCH MODEL STATS:")
    print(f"Train Loss History : {loss_history_pytorch.get("train_loss")}")
    print(f"Val Loss History : {loss_history_pytorch.get("val_loss")}")
    print(f"Min Train Loss: {min_train_loss_pytorch}")
    print(f"Min Val Loss: {min_val_loss_pytorch}")

    print("SCRATCH MODEL STATS:")
    print(f"Train Loss History : {loss_history_scratch.get("train_loss")}")
    print(f"Val Loss History : {loss_history_scratch.get("val_loss")}")
    print(f"Min Train Loss: {min_train_loss_scratch}")
    print(f"Min Val Loss: {min_val_loss_scratch}")
