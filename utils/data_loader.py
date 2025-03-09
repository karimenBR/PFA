from torchvision import transforms
from medmnist import ChestMNIST
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = ChestMNIST(split='train', transform=transform, download=True)
    val_dataset = ChestMNIST(split='val', transform=transform, download=True)
    test_dataset = ChestMNIST(split='test', transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader