import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import roc_auc_score  # For AUC calculation
import numpy as np
from Model.resnet18_model import ResNet
from utils.data_loader import get_data_loaders
from utils.visualize import visualize_predictions

#torch._dynamo.config.suppress_errors = True  # Disable Dynamo errors
def getAUC(y_true, y_score, task):
    """AUC metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    """
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == "multi-label, binary-class":
        auc = 0
        for i in range(y_score.shape[1]):
            label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
            auc += label_auc
        ret = auc / y_score.shape[1]
    elif task == "binary-class":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = roc_auc_score(y_true, y_score)
    else:
        auc = 0
        for i in range(y_score.shape[1]):
            y_true_binary = (y_true == i).astype(float)
            y_score_binary = y_score[:, i]
            auc += roc_auc_score(y_true_binary, y_score_binary)
        ret = auc / y_score.shape[1]

    return ret

def train_model(num_epochs=10, batch_size=64, learning_rate=0.001):
    print("Initializing training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size)
    print("Data loaded successfully.")

    # Initialize model, loss, and optimizer
    print("Initializing model...")
    num_classes = len(train_loader.dataset.info['label'])
    model = ResNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Weight decay

    # Define milestones and gamma for MultiStepLR
    milestones = [3, 6, 9]  # Epochs at which to reduce the learning rate
    gamma = 0.1  # Factor by which the learning rate is reduced
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)  # MultiStepLR scheduler

    print("Model initialized.")

    # Early stopping
    best_val_accuracy = 0
    patience = 10  # Number of epochs to wait for improvement
    patience_counter = 0

    # Training loop
    print("Starting training loop...")
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Fix labels
            if labels.dtype != torch.long:
                labels = labels.long()
            if len(labels.shape) > 1:
                labels = torch.argmax(labels, dim=1)
            labels = labels.squeeze()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                # Fix labels
                if labels.dtype != torch.long:
                    labels = labels.long()
                if len(labels.shape) > 1:
                    labels = torch.argmax(labels, dim=1)
                labels = labels.squeeze()

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Collect all labels and predictions for AUC calculation
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.softmax(dim=1).cpu().numpy())

        # Calculate validation accuracy
        val_accuracy = 100 * correct / total
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        # Calculate AUC
        auc = getAUC(np.array(all_labels), np.array(all_preds), task="multi-class")  # Use the task type of your dataset
        print(f"Validation AUC: {auc:.4f}")

        # Early stopping
        #if val_accuracy > best_val_accuracy:
         #   best_val_accuracy = val_accuracy
         #   patience_counter = 0  # Reset patience counter
         #   torch.save(model.state_dict(), "saved_models/best_chestmnist_resnet.pth")  # Save best model
        #else:
           # patience_counter += 1
            #if patience_counter >= patience:
              #  print("Early stopping triggered.")
               # break

        # Step the scheduler
        scheduler.step()

    # Save the final model
    print("Saving final model...")
    torch.save(model.state_dict(), "saved_models/chestmnist_resnet.pth")
    print("Model saved.")

    # Visualize predictions
    print("Visualizing predictions...")
    visualize_predictions(model, test_loader, device)
    print("Training complete.")

if __name__ == "__main__":
    train_model()