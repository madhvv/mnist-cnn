# src/data.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(batch_size=64, valid_ratio=0.1, data_dir="data"):
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.07,0.07), scale=(0.95,1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_full = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_train)
    test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_test)

    n_valid = int(len(train_full) * valid_ratio)
    if n_valid == 0:
        train_set = train_full
        valid_set = None
    else:
        n_train = len(train_full) - n_valid
        train_set, valid_set = random_split(train_full, [n_train, n_valid])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2) if valid_set else None
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, valid_loader, test_loader
