import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms

from data_setup import create_dataloaders
from model_builder import EfficientNetB0


def save_model(model: torch.nn.Module, targ_dir: str, model_name: str):
    target_dir_path = Path(targ_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

def accuracy_fn(y_pred: torch.Tensor, y_true: torch.Tensor):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred))
    return acc

def predict(image):

    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
    ])

    _, _, _, classes = create_dataloaders("data/train", "data/test", 32, transform, transform)

    model = EfficientNetB0(in_channels=3, out_channels=len(classes)).to("cpu")

    model.load_state_dict(torch.load("models/efficient_net_b0.pth", weights_only=True, map_location=torch.device('cpu')))

    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0)

    model.eval()
    with torch.inference_mode():
        output = model(image)
    predicted_class = torch.argmax(output, dim=1).item()

    return f"Predicted Class: {classes[predicted_class]}"
