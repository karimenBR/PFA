from medmnist import ChestMNIST, INFO
from torchvision import transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=32, img_size=224, download=True):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    train_dataset = ChestMNIST(split='train', transform=transform, download=download)
    val_dataset   = ChestMNIST(split='val', transform=transform, download=download)
    test_dataset  = ChestMNIST(split='test', transform=transform, download=download)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader