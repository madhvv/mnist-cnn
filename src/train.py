# src/train.py
import argparse, os
import torch, torch.nn as nn, torch.optim as optim
from src.data import get_dataloaders
from src.model import SimpleMNISTCNN
from tqdm import tqdm

def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()
    return 100.0 * correct / total

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size, valid_ratio=0.1)
    model = SimpleMNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}")
        for x,y in loop:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

        if val_loader:
            val_acc = evaluate(model, val_loader, device)
            print(f"Epoch {epoch} Val Acc: {val_acc:.2f}%")
            if val_acc > best_val:
                best_val = val_acc
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), "models/best.pth")
    print("Done. Best val acc:", best_val)
    if test_loader:
        test_acc = evaluate(model, test_loader, device)
        print("Test accuracy:", test_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
