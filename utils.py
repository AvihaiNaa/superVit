import torch
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader



def load_dataset(dataset_name="MNIST", batch_size=128):
    if dataset_name == "MNIST":
        from torchvision.datasets.mnist import MNIST
        transform = ToTensor()
        train_set = MNIST(root='./../datasets', train=True, download=True, transform=transform)
        test_set = MNIST(root='./../datasets', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)
    train_loader, validation_loader = get_train_and_validation_loaders(train_loader,
                                                                           validation_split=0.1,
                                                                           batch_size=batch_size)
    return train_loader, validation_loader, test_loader



def get_train_and_validation_loaders(dataloader, validation_split=0.1, batch_size=32, shuffle=True, rand_seed=42):
    import numpy as np
    from torch.utils.data.sampler import SubsetRandomSampler
    '''
    Inspired by:https://stackoverflow.com/a/50544887
    Args:
        dataloader (torch DataLoader): dataloader torch type
        validation_split (float): size of validation set out of the original train set. Default is 0.1
        batch_size (int): batch size. Default is 32.
        shuffle (bool): default if True.
        rand_seed (int): random seed for shuffling. default is 42

    Returns:
        train_loader, validation_loader
    '''
    # Creating data indices for training and validation splits:
    dataset_size = len(dataloader.dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle:
        np.random.seed(rand_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    return train_loader, validation_loader