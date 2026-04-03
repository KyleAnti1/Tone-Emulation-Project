import torch.nn as nn
import torch.optim as optim

def get_loss(loss_type="mse"):

    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "l1":
        return nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def get_optimizer(model, lr=1e-3):
    return optim.Adam(model.parameters(), lr=lr)

def train(model, dataloader, loss_fn, optimizer, epochs, device="cpu"):

    model.to(device)

    for epoch in range(epochs):

        model.train()  # set model to training mode
        total_loss = 0.0

        for batch_idx, (x, y) in enumerate(dataloader):

            # Move data to device
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            pred = model(x)

            # Compute loss
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")