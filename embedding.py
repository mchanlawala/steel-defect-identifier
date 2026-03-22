import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load ResNet50 model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove last layer
model.eval()

# Image transform with normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_embedding(image_path):
    """
    Extract embedding from image and return as numpy array
    """
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            embedding = model(image)

        # Convert to numpy and flatten
        embedding = embedding.squeeze().numpy()

        # Ensure it's float32 (FAISS requires this)
        embedding = embedding.astype(np.float32)

        return embedding

    except Exception as e:
        print(f"Error extracting embedding from {image_path}: {e}")
        # Return zero vector as fallback
        return np.zeros(2048, dtype=np.float32)