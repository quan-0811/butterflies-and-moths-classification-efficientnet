import os
import torch
import torch.nn as nn
from torchvision import transforms
from model_builder import EfficientNetB0
from data_setup import create_dataloaders
from engine import train
from utils import save_model, accuracy_fn

# Hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.005

# Directories
train_dir = "data/train"
valid_dir = "data/valid"
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

# Load model checkpoint
model_path = "models/efficient_net_b0.pth"
if os.path.exists(model_path):
    try:
        efficient_net_b0.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
else:
    print("No pre-trained model found. Initializing a new model.")

# Loss function, accuracy function and optimizer
loss_fn = nn.CrossEntropyLoss()
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
