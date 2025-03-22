import torch
from torchvision import transforms
from PIL import Image
from Model.resnet_model import ResNet

def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = transform(image).unsqueeze(0).to(device)

    # Load the trained model
    model = ResNet(num_classes=14).to(device)  # Use ResNet instead of CNN
    model.load_state_dict(torch.load("saved_models/chestmnist_resnet.pth"))  # Updated model name

    # Make a prediction
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        print(f"Predicted class: {predicted.item()}")

if __name__ == "__main__":
    predict("path_to_your_image.png")