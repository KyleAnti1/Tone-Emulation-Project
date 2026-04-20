import torch.nn as nn
import torch.optim as optim
import torch


def get_loss(loss_type="mse"):
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "l1":
        return nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def get_optimizer(model, lr=1e-3):
    return optim.Adam(model.parameters(), lr=lr)


def train(model, train_loader, val_loader, loss_fn, optimizer, epochs, device="cpu"):
    model.to(device)

    train_losses = []
    val_losses   = []

    for epoch in range(epochs):

        # Training
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item()

        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f}")

    return train_losses, val_losses