# src/utils.py
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_training(losses, accs):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label='Loss')
    plt.plot(accs, label='Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Training Performance')
    plt.show()

def evaluate_model(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            labels.extend(y.cpu().numpy())

    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print(classification_report(labels, preds))
