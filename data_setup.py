import torchvision
from torch.utils.data import DataLoader, random_split


def create_dataloaders(train_dir: str, test_dir: str, batch_size: int, train_transform, valid_test_transform):

    train_dataset = torchvision.datasets.ImageFolder(root=train_dir,
                                                     transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_dir,
                                                    transform=valid_test_transform)

    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size

    _, valid_dataset = random_split(train_dataset, [train_size, valid_size])

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=True)

    return train_dataloader, valid_dataloader, test_dataloader, train_dataset.classes