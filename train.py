import torch
import torch.nn as nn
import torch.optim as optim
from Model.resnet_model  import ResNet  # Updated import
from utils.data_loader import get_data_loaders
from utils.visualize import visualize_predictions
def train_model(num_epochs=10, batch_size=64, learning_rate=0.001):
    print("Initializing training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size)
    print("Data loaded successfully.")

    # Initialize model, loss, and optimizer
    print("Initializing model...")
    num_classes = len(train_loader.dataset.info['label'])
    model = ResNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Model initialized.")

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
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
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

        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

    # Save the model
    print("Saving model...")
    torch.save(model.state_dict(), "saved_models/chestmnist_resnet.pth")
    print("Model saved.")

    # Visualize predictions
    print("Visualizing predictions...")
    visualize_predictions(model, test_loader, device)
    print("Training complete.")