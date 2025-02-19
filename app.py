import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
import os
from data_setup import create_dataloaders
from model_builder import EfficientNetB0

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to model's input size
    transforms.ToTensor(),          # Convert image to tensor
])

_, _, _, classes = create_dataloaders("data/train", "data/valid", "data/test", 32, transform)

efficient_net_b0 = EfficientNetB0(in_channels=3, out_channels=len(classes)).to("cpu")

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

# Define prediction function
def predict(image):
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.inference_mode():
        output = efficient_net_b0(image)
    predicted_class = torch.argmax(output, dim=1).item()
    return f"Predicted Class: {classes[predicted_class]}"

# Create Gradio interface
iface = gr.Interface(fn=predict, inputs="image", outputs="text")
iface.launch()
