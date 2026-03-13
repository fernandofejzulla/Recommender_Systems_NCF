import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

#one full training epoch
def train_one_epoch(model, dataloader, optimizer, criterion, device):

    model.train()
    total_loss=0

    for batch in tqdm(dataloader):

        users = batch["user"].to(device)
        items = batch["item"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        predictions = model(users, items)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

#validation function
#no gradient upd
#eval mode
def validate(model, dataloader, criterion, device):

    model.eval()
    total_loss=0

    with torch.no_grad():
        for batch in dataloader:
            users = batch["user"].to(device)
            items = batch["item"].to(device)
            labels = batch["label"].to(device)

            predictions = model(users, items)
            loss = criterion(predictions, labels)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

#training loop
#early stopping for tracking the best validation loss
def train_model(model, train_dataset, val_dataset, epochs=20, batch_size=256, learning_rate=0.001, patience=3, device="cpu"):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    train_losses = []
    val_losses = []

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    patience_counter = 0

    model.to(device)

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        train_losses.append(train_loss) 
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_[128_64_32].pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping!!")
            break

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.savefig("loss_curve_[128_64_32].png")

    print("Training done!!")

