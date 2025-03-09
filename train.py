import torch
import torch.nn as nn
import torch.optim as optim
from Model.cnn_model import CNN
from utils.data_loader import get_data_loaders
from utils.visualize import visualize_predictions

def train_model(num_epochs=10, batch_size=64, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, test_loader = get_data_loaders(batch_size)

    # Initialize model, loss, and optimizer
    num_classes = len(train_loader.dataset.info['label'])
    model = CNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.squeeze()).sum().item()

        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

    # Save the model
    torch.save(model.state_dict(), "saved_models/chestmnist_cnn.pth")

    # Visualize predictions
    visualize_predictions(model, test_loader, device)

if __name__ == "__main__":
    train_model()