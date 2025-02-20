import torch
from pathlib import Path
import torchvision.datasets
from PIL import Image
from torchvision import transforms
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

    classes = torchvision.datasets.ImageFolder(root="data/train", transform=transform).classes

    model = EfficientNetB0(in_channels=3, out_channels=len(classes)).to("cpu")

    model.load_state_dict(torch.load("models/efficient_net_b0.pth", weights_only=True, map_location=torch.device('cpu')))

    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0)

    model.eval()
    with torch.inference_mode():
        output = model(image)

    probabilities = torch.softmax(output, dim=1)[0]

    top_probs, top_indices = torch.topk(probabilities, 7)  # Get top 3 classes
    top_classes = [classes[idx] for idx in top_indices.tolist()]
    top_probs = top_probs.tolist()

    results = {top_classes[i]: float(top_probs[i]) for i in range(len(top_classes))}

    return results
