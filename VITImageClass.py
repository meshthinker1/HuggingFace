from PIL import Image
import requests
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import ViTFeatureExtractor, ViTForImageClassification

# Load pre-trained ViT model and feature extractor
model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Define image preprocessing pipeline
preprocess = Compose([
    Resize(224),      # Resize image to 224x224 pixels
    CenterCrop(224),  # Crop the center of the image
    ToTensor(),       # Convert image to PyTorch tensor
    Normalize(        # Normalize image using pre-computed mean and standard deviation
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# Function to classify an image
def classify_image(image_path):
    # Load and preprocess image
    image = Image.open(image_path)
    inputs = preprocess(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(**feature_extractor(images=inputs))

    # Get predicted class label
    predicted_class_idx = torch.argmax(outputs.logits, dim=-1).item()

    # Get class labels
    labels = requests.get("https://raw.githubusercontent.com/huggingface/datasets/1.11.0/metrics/imagenet2012_label_map.txt").text.split("\n")

    # Return predicted class label
    return labels[predicted_class_idx]

# Example usage
image_path = "example_image.jpg"
predicted_class = classify_image(image_path)
print("Predicted class:", predicted_class)
