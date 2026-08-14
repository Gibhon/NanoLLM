import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import torch
from tqdm import tqdm


def train_and_val(
    model,
    epochs,
    loss_fn,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    device,
    model_path,
):
    start = time.time()
    loss_history = {"train_loss": [], "val_loss": []}
    min_train_loss = float("inf")
    min_val_loss = float("inf")

    # tqdm progress tracker
    total_steps = len(train_loader) * epochs

    # Train Loop
    with tqdm(total=total_steps, desc="Training NanoLLM", unit="batch") as pbar:
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            for x_batch, y_batch in train_loader:
                x, y = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad(set_to_none=True)

                logits = model(x)
                loss = loss_fn(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )  # [batch_size*block_size , vocab_len] and [block_size * batch_size]
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                total_train_loss += loss.item()

                pbar.update(1)
                pbar.set_postfix(
                    epoch=f"{epoch + 1}/{epochs}",
                    batch_loss=f"{loss.item():.4f}",
                )

            avg_train_loss = total_train_loss / len(train_loader)
            loss_history["train_loss"].append(avg_train_loss)
            if avg_train_loss < min_train_loss:
                min_train_loss = avg_train_loss

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x, y = x_batch.to(device), y_batch.to(device)
                    logits = model(x)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                    total_val_loss += loss.item()
                avg_val_loss = total_val_loss / len(val_loader)
                loss_history["val_loss"].append(avg_val_loss)
                if avg_val_loss < min_val_loss:
                    min_val_loss = avg_val_loss
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": min_val_loss,
                        },
                        model_path,
                    )
            pbar.set_postfix(
                epoch=f"{epoch + 1}/{epochs}",
                train_loss=f"{avg_train_loss:.4f}",
                val_loss=f"{avg_val_loss:.4f}",
                best_val=f"{min_val_loss:.4f}",
            )
    total_time = time.time() - start
    return loss_history, min_train_loss, min_val_loss, total_time


def plot_model_comparison(
    loss_history_pytorch,
    min_train_loss_pytorch,
    min_val_loss_pytorch,
    time_pytorch,
    loss_history_scratch,
    min_train_loss_scratch,
    min_val_loss_scratch,
    time_scratch,
    graph_path
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Model Performance Comparison: PyTorch Built-in vs. Custom Scratch",
        fontsize=16,
        fontweight="bold",
        y=1.05,
    )

    epochs = range(1, len(loss_history_pytorch["train_loss"]) + 1)
    ax1 = axes[0]

    # PyTorch lines (Blue hues)
    ax1.plot(
        epochs,
        loss_history_pytorch["train_loss"],
        label="PyTorch Train",
        color="#1f77b4",
        linestyle="--",
        marker="o",
        markersize=4,
    )
    ax1.plot(
        epochs,
        loss_history_pytorch["val_loss"],
        label="PyTorch Val",
        color="#005b96",
        linestyle="-",
        marker="s",
        markersize=4,
    )

    ax1.plot(
        epochs,
        loss_history_scratch["train_loss"],
        label="Scratch Train",
        color="#ff7f0e",
        linestyle="--",
        marker="o",
        markersize=4,
    )
    ax1.plot(
        epochs,
        loss_history_scratch["val_loss"],
        label="Scratch Val",
        color="#cc4f00",
        linestyle="-",
        marker="s",
        markersize=4,
    )

    ax1.set_title("Loss History per Epoch", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.7)

    ax2 = axes[1]
    models = ["PyTorch", "Scratch"]
    times = [time_pytorch, time_scratch]
    colors = ["#1f77b4", "#ff7f0e"]

    bars = ax2.bar(models, times, color=colors, width=0.5)

    ax2.set_title("Training Speed Comparison", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Time (seconds)")
    ax2.grid(axis="y", linestyle=":", alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            yval + (max(times) * 0.02),
            f"{yval:.2f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax3 = axes[2]

    x = np.arange(len(models))
    width = 0.35

    min_train_losses = [min_train_loss_pytorch, min_train_loss_scratch]
    min_val_losses = [min_val_loss_pytorch, min_val_loss_scratch]

    rects1 = ax3.bar(
        x - width / 2, min_train_losses, width, label="Min Train Loss", color="#2ca02c"
    )
    rects2 = ax3.bar(
        x + width / 2, min_val_losses, width, label="Min Val Loss", color="#d62728"
    )

    ax3.set_title("Best Loss Metrics", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Loss")
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.legend()
    ax3.grid(axis="y", linestyle=":", alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax3.annotate(
                f"{height:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(
        graph_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
