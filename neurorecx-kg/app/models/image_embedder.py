from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

class ImageEmbedder:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def embed(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        return outputs.squeeze().numpy()
