import torch
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F

import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent / "src"

sys.path.append(str(SRC_PATH))
from train import train_and_val  # type: ignore

sys.path.append(str(SRC_PATH / "models"))
from pytorch_model import NanoLLM  # type: ignore

# -------Scratch Model-----------

sys.path.append(str(SRC_PATH / "data_manager"))
from data_cleaner import clean_data  # type: ignore
from tokenizer import CharTokenizer  # type: ignore
from dataset import NanoLLMDataset  # type: ignore

if __name__ == "__main__":
    # ---Config---
    EMBEDDING_DIMENSION = 128
    NUM_ATTENTION_HEADS = 4
    NUM_LAYERS = 4
    NUM_FEEDFORWARD = EMBEDDING_DIMENSION * 4
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    BLOCK_SIZE = 128

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 6
    LR = 3e-4
    DROPOUT = 0.2
    WEIGHT_DECAY = 1e-3

    DATA_PATH = Path(__file__).resolve().parent / "data"

    # ---Data---
    data = clean_data(DATA_PATH / "data.txt")
    tokenizer = CharTokenizer(data)
    token_ids = tokenizer.encode(data)

    dataset = NanoLLMDataset(token_ids, BLOCK_SIZE)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset=dataset, lengths=[train_size, val_size]
    )

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
    # best_val=0.9620, train_loss=1.6466, val_loss=0.9620
    shakespeare_model_pytorch = NanoLLM(
        tokenizer,
        embed_dim=EMBEDDING_DIMENSION,
        n_heads=NUM_ATTENTION_HEADS,
        n_layers=NUM_LAYERS,
        n_feedforward=NUM_FEEDFORWARD,
        block_size = BLOCK_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)

    # ---Model Necessities---
    optimizer = torch.optim.AdamW(
        shakespeare_model_pytorch.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    loss_fn = F.cross_entropy

    # ---Train---
    model_path = DATA_PATH / "pytorch_model.pth"
    loss_history, min_train_loss, min_val_loss = train_and_val(
        model=shakespeare_model_pytorch,
        epochs=EPOCHS,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        model_path=model_path,
    )
