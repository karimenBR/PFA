from torchvision import transforms
from medmnist import ChestMNIST
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64):
    # Define transformations for training data (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),  # Convert grayscale to RGB
        transforms.RandomHorizontalFlip(),           # Randomly flip images
        transforms.RandomRotation(10),              # Randomly rotate images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Define transformations for validation and test data (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),  # Convert grayscale to RGB
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load the datasets
    train_dataset = ChestMNIST(split='train', transform=train_transform, download=True)
    val_dataset = ChestMNIST(split='val', transform=val_transform, download=True)
    test_dataset = ChestMNIST(split='test', transform=val_transform, download=True)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader