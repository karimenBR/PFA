import torch
from Model.resnet18_model  import ResNet  # Updated import
from utils.data_loader import get_data_loaders

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    _, _, test_loader = get_data_loaders()

    # Load the trained model
    num_classes = len(test_loader.dataset.info['label'])
    model = ResNet(num_classes=num_classes).to(device)  # Use ResNet instead of CNN
    model.load_state_dict(torch.load("saved_models/chestmnist_resnet.pth"))  # Updated model name

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    test_model()