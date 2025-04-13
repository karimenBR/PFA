import torch
from Model.rasnet34 import ResNet
from utils.data_loader import get_data_loaders
from medmnist import INFO
import numpy as np

def test(model, test_loader, device, threshold=0.5):
    model.eval()
    total = 0
    correct = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float()

            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()

            correct += (preds == labels).sum().item()
            total += labels.numel()

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    accuracy = 100 * correct / total
    print(f"🔍 Test Accuracy (per label): {accuracy:.2f}%")

    # Optional: Return for other metrics (F1, AUC)
    return torch.cat(all_labels), torch.cat(all_preds)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_data_loaders(batch_size=32)

    # Load your trained model (make sure to save it first!)
    model = ResNet(num_classes=14)
    model.load_state_dict(torch.load("C:\\Users\karim\PycharmProjects\PFA\saved_models\chestmnist_resnet34.pth", map_location=device))
    model.to(device)

    test(model, test_loader, device)