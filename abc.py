
import torch
import torchvision
from torchvision import transforms
from data_setup import create_dataloaders

NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.005

# Directories
train_dir = "data1/train"
valid_dir = "data1/val"
test_dir = "data1/test"

transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=transforms)

print(train_dataset.class_to_idx)