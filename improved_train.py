import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from torch.cuda.amp import GradScaler, autocast
import time
from collections import defaultdict

from Model.improved_RazNet import ImprovedResNet
from utils.data_loader import get_data_loaders

"""alech el early stopping : baad num mou3ayen ml epoch itha kan el accuracy didn't improve ywa9f el training """
class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='best_model.pth', metric='val_auc', mode='max'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.metric = metric
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(model)
        elif (self.mode == 'max' and current_score < self.best_score + self.delta) or \
                (self.mode == 'min' and current_score > self.best_score - self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


def calculate_metrics(y_true, y_pred, y_score):
    """Calculate metrics for multi-label classification"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    metrics = {
        'accuracy': 0.0,
        'auc': 0.0,
        'ap': 0.0  # Average Precision
    }

    # Calculate per-class metrics
    n_classes = y_true.shape[1]
    for i in range(n_classes):
        try:
            metrics['auc'] += roc_auc_score(y_true[:, i], y_score[:, i])
            metrics['ap'] += average_precision_score(y_true[:, i], y_score[:, i])
        except ValueError:
            continue

    # Calculate accuracy (strict)
    metrics['accuracy'] = np.all(y_pred == y_true, axis=1).mean()

    # Average metrics
    metrics['auc'] /= n_classes
    metrics['ap'] /= n_classes

    return metrics


def train_model(num_epochs=50, batch_size=64, base_lr=0.0001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Mixed precision training
    scaler = GradScaler()

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size)

    # Get number of classes - ChestMNIST specific
    num_classes = len(train_loader.dataset.info['label'])
    print(f"Number of classes: {num_classes}")

    # Initialize model - REMOVED SIGMOID for BCEWithLogitsLoss
    print("Initializing model...")
    model = ImprovedResNet(num_classes=num_classes).to(device)

    # Loss and optimizer - Changed to BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()  # Changed from BCELoss
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    early_stopping = EarlyStopping(patience=10, metric='val_auc')

    # Training history
    history = defaultdict(list)

    print("\nStarting training...")
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        # Training phase
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device).float()

            # Mixed precision forward pass
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)

            # Print batch progress
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch + 1} | Batch {batch_idx + 1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")

        # Calculate epoch training loss
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                # Apply sigmoid for predictions and metrics
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                all_preds.append(preds.cpu())
                all_scores.append(probs.cpu())
                all_labels.append(labels.cpu())

        # Calculate validation metrics
        val_loss /= len(val_loader.dataset)
        val_preds = torch.cat(all_preds).numpy()
        val_scores = torch.cat(all_scores).numpy()
        val_labels = torch.cat(all_labels).numpy()

        metrics = calculate_metrics(val_labels, val_preds, val_scores)

        # Print epoch summary
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch + 1}/{num_epochs} | Time: {epoch_time:.1f}s")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {metrics['accuracy']:.4f}")
        print(f"Val AUC: {metrics['auc']:.4f}")
        print(f"Val AP: {metrics['ap']:.4f}")

        # Early stopping check
        early_stopping(metrics['auc'], model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        scheduler.step(metrics['auc'])

    # Load best model weights
    model.load_state_dict(torch.load('best_model.pth'))

    # Test phase
    model.eval()
    test_preds = []
    test_scores = []
    test_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float()

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            test_preds.append(preds.cpu())
            test_scores.append(probs.cpu())
            test_labels.append(labels.cpu())

    # Calculate test metrics
    test_preds = torch.cat(test_preds).numpy()
    test_scores = torch.cat(test_scores).numpy()
    test_labels = torch.cat(test_labels).numpy()

    test_metrics = calculate_metrics(test_labels, test_preds, test_scores)

    print("\n=== Final Test Results ===")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    print(f"Test AP: {test_metrics['ap']:.4f}")

    return model, history


if __name__ == "__main__":
    model, history = train_model()