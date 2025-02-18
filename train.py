import torch
import torch.nn as nn
import torchmetrics
from torchvision import transforms
from model_builder import EfficientNetB0
from data_setup import create_dataloaders
from engine import train
from utils import save_model

# Hyperparameters
NUM_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Directories
train_dir = "data/train"
valid_dir = "data/test"
test_dir = "data/test"

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Transforms
data_transforms = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor()
])

# Create dataloaders
train_dataloader, valid_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                         valid_dir=valid_dir,
                                                                         test_dir=test_dir,
                                                                         batch_size=BATCH_SIZE,
                                                                         transform=data_transforms)

# Create model
efficient_net_b0 = EfficientNetB0(in_channels=3, out_channels=len(class_names)).to(device)

# Loss function, accuracy function and optimizer
loss_fn = nn.CrossEntropyLoss()
accuracy_fn = torchmetrics.Accuracy(task="multiclass", num_classes=len(class_names)).to(device)
optimizer = torch.optim.RMSprop(efficient_net_b0.parameters(), lr=LEARNING_RATE)

# Train the model
train(num_epochs=NUM_EPOCHS,
      model=efficient_net_b0,
      train_dataloader=train_dataloader,
      valid_dataloader=valid_dataloader,
      loss_fn=loss_fn,
      optimizer=optimizer,
      accuracy_fn=accuracy_fn,
      device=device)

save_model(model=efficient_net_b0,
                 targ_dir="models",
                 model_name="efficient_net_b0.pth")
