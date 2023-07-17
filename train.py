import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from config import CONFIG
from models.train_model import train as train_model


def train_vit():
    train_loader, test_loader = load_dataset(dataset_name="MNIST", batch_size=CONFIG.VIT.BATCH_SIZE)
    model = train_model(train_loader, test_loader)
    print("ansqnl")



def load_dataset(dataset_name="MNIST", batch_size=128):
    if dataset_name == "MNIST":
        from torchvision.datasets.mnist import MNIST
        transform = ToTensor()
        train_set = MNIST(root='./../datasets', train=True, download=True, transform=transform)
        test_set = MNIST(root='./../datasets', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)
    return train_loader, test_loader


if __name__ == "__main__":
    train_vit()