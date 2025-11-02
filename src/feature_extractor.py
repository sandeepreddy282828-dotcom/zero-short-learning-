# feature_extractor.py

import torch
from transformers import CLIPTokenizer, CLIPModel
import numpy as np

def extract_class_vectors(class_file, output_file='class_vectors.npy'):
    # Load CLIP model and tokenizer
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    with open(class_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    # Tokenize and encode
    inputs = tokenizer(class_names, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        class_vectors = outputs.cpu().numpy()

    # Save class vectors and class names
    np.save('class_vectors.npy', class_vectors)
    with open('train_classes.txt', 'w') as f:
        f.writelines("\n".join(class_names))

    print("✅ Class vectors saved to class_vectors.npy")

if __name__ == "__main__":
    extract_class_vectors("train_classes.txt")
