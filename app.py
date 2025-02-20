import gradio as gr
from utils import predict

iface = gr.Interface(
    fn=predict,
    inputs="image",
    outputs=gr.Label(num_top_classes=7),
    title="EfficientNet-B0 Image Classification",
    description="Upload an image and see the top 3 predicted classes with probabilities."
)

iface.launch()