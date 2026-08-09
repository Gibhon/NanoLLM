import torch
from tqdm import tqdm


def train_and_val(
    model,
    epochs,
    loss_fn,
    optimizer,
    train_loader,
    val_loader,
    device,
    model_path,
):
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

    return loss_history, min_train_loss, min_val_loss
