import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

def create_dataloaders(train_dir: str, valid_dir: str, test_dir: str, batch_size: int, transform):

    train_dataset = torchvision.datasets.ImageFolder(root=train_dir,
                                                     transform=transform)
    valid_dataset = torchvision.datasets.ImageFolder(root=valid_dir,
                                                     transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_dir,
                                                    transform=transform)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size)

    return train_dataloader, valid_dataloader, test_dataloader, train_dataset.classes

if __name__ == "__main__":
    train_dir = "data/train"
    valid_dir = "data/valid"
    test_dir = "data/test"
    transforms = transforms.ToTensor()
    _, _, _, classes = create_dataloaders(train_dir, valid_dir, test_dir, 32, transforms)
    print(classes)