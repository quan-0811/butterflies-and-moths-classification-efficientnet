from torchvision import datasets, transforms
from torch.utils.data import Dataset, ConcatDataset
from typing import List

class AugmentedDataset(Dataset):
    def __init__(self, root, transform: transforms.Compose, aug_transforms: List[transforms.Compose]):

        self.original_dataset = datasets.ImageFolder(root=root, transform=transform)
        self.augmented_datasets = [
            datasets.ImageFolder(root=root, transform=aug_transform) for aug_transform in aug_transforms
        ]

        self.full_dataset = ConcatDataset([self.original_dataset] + self.augmented_datasets)

    def __len__(self):
        return len(self.full_dataset)

    def __getitem__(self, idx):
        return self.full_dataset[idx]

