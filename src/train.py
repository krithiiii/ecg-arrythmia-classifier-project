# src/train.py
import torch
from torch.utils.data import DataLoader, random_split
from model import ECG1DCNN
from data_loader import MITBIHDataset
from utils import plot_training, evaluate_model

def train_model():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = MITBIHDataset(record_ids=['100', '101', '102'])  # few records to start
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # Model
    model = ECG1DCNN(num_classes=5).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training
    num_epochs = 10
    train_losses, train_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {running_loss:.4f} | Accuracy: {acc:.2f}")
        train_losses.append(running_loss)
        train_accs.append(acc)

    # Results
    plot_training(train_losses, train_accs)
    evaluate_model(model, val_loader, device)

if __name__ == "__main__":
    train_model()