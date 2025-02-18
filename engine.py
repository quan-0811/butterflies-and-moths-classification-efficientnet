import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

def train_step(model: nn.Module, train_dataloader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, accuracy_fn, device: torch.device):
    model.train()

    total_train_loss, total_train_accuracy = 0, 0

    for batch, (X_train, y_train) in enumerate(train_dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)

        # Forward pass
        y_train_logit = model(X_train)
        y_train_pred = torch.softmax(y_train_logit, dim=0).argmax(dim=1)

        # Loss & accuracy calculation
        loss = loss_fn(y_train_logit, y_train)
        total_train_loss += loss
        total_train_accuracy += accuracy_fn(y_train_pred, y_train) * 100

        # Backpropagation
        loss.backward()

        optimizer.zero_grad()
        optimizer.step()

    avg_loss_per_batch = total_train_loss / len(train_dataloader)
    avg_acc_per_batch = total_train_accuracy / len(train_dataloader)

    return avg_loss_per_batch, avg_acc_per_batch

def valid_test_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, accuracy_fn, device: torch.device):
    model.eval()

    with torch.inference_mode():
        total_loss, total_accuracy = 0, 0

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Forward pass
            y_logit = model(X)
            y_pred = torch.softmax(y_logit, dim=0).argmax(dim=1)

            # Loss & accuracy calculation
            loss = loss_fn(y_logit, y)
            total_loss += loss
            total_accuracy += accuracy_fn(y_pred, y) * 100

        avg_loss_per_batch = total_loss / len(dataloader)
        avg_acc_per_batch = total_accuracy / len(dataloader)

    return avg_loss_per_batch, avg_acc_per_batch

def train(num_epochs: int, model: nn.Module, train_dataloader: DataLoader, valid_dataloader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, accuracy_fn, device: str):
    results = {"train_loss": [],
               "train_acc": [],
               "valid_loss": [],
               "valid_acc": []
               }

    model.to(device)

    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch: {epoch}\n----------")
        avg_train_loss, avg_train_acc = train_step(model, train_dataloader, loss_fn, optimizer, accuracy_fn, device)
        print(f"Train Loss: {avg_train_loss:.5f} | Train Accuracy: {avg_train_acc:.4f}%")
        avg_valid_loss, avg_valid_acc = valid_test_step(model, valid_dataloader, loss_fn, accuracy_fn, device)
        print(f"Valid Loss: {avg_valid_loss:.5f} | Valid Accuracy: {avg_valid_acc:.4f}%")

        # Update results dictionary
        results["train_loss"].append(avg_train_loss)
        results["train_acc"].append(avg_train_acc)
        results["valid_loss"].append(avg_valid_loss)
        results["valid_acc"].append(avg_valid_acc)

    return results


