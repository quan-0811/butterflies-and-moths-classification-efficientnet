import os
import torch
import torch.nn as nn
from torchvision import transforms
from model_builder import EfficientNetB0
from data_setup import create_dataloaders
from engine import train
from utils import save_model, accuracy_fn

NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.005

train_dir = "data/train"
valid_dir = "data/valid"
test_dir = "data/test"

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
])

augmentation_transforms = [
    transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ]),
    transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
    ]),
    transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.2),
        transforms.RandomResizedCrop(size=224, scale=(0.95, 1.0)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
    ])
]

train_dataloader, valid_dataloader, test_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                                      valid_dir=valid_dir,
                                                                                      test_dir=test_dir,
                                                                                      batch_size=BATCH_SIZE,
                                                                                      transform=transform,
                                                                                      augmented_transforms=augmentation_transforms)

efficient_net_b0 = EfficientNetB0(in_channels=3, out_channels=len(class_names)).to(device)

model_path = "models/efficient_net_b0.pth"
if os.path.exists(model_path):
    try:
        efficient_net_b0.load_state_dict(torch.load(model_path, weights_only=True))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
else:
    print("No pre-trained model found. Initializing a new model.")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(efficient_net_b0.parameters(), lr=LEARNING_RATE)

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
