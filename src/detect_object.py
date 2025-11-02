# detect_object.py

import sys
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

def load_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4815, 0.4578, 0.4082), std=(0.2686, 0.2613, 0.2758))
    ])
    img = Image.open(image_path).convert("RGB")
    return preprocess(img).unsqueeze(0)

def main(image_path):
    # Load model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    # Load image and get embedding
    image_tensor = load_image(image_path)
    with torch.no_grad():
        image_features = model.get_image_features(image_tensor)

    # Load class vectors
    class_vectors = np.load('class_vectors.npy')
    class_names = [line.strip() for line in open('train_classes.txt').readlines()]

    # Compute cosine similarity
    similarities = cosine_similarity(image_features.cpu().numpy(), class_vectors)[0]
    top_indices = similarities.argsort()[-5:][::-1]

    print("\n--- Top-5 Prediction ---")
    for i in top_indices:
        print(f"{class_names[i]:<15} similarity={similarities[i]:.4f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python detect_object.py <image_path>")
        sys.exit(1)
    main(sys.argv[1])
