import torch
import torch.nn as nn
import torch.optim as optim
from Model.rasnet34 import ResNet
from utils.data_loader import get_data_loaders
from medmnist import INFO


def train(model, train_loader, val_loader, device, num_epochs=10, lr=0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")
        evaluate(model, val_loader, device)

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            outputs = model(images)
            predicted = torch.sigmoid(outputs) > 0.5
            correct += (predicted == labels.bool()).sum().item()
            total += labels.numel()

    print(f"Validation Accuracy (per label): {(100 * correct / total):.2f}%")
    torch.save(model.state_dict(), "saved_models/chestmnist_resnet34.pth")

if __name__ == "__main__":
    num_epochs = 10
    batch_size = 32
    num_classes = 14
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_data_loaders(batch_size=batch_size)
    model = ResNet(num_classes)
    train(model, train_loader, val_loader, device, num_epochs)