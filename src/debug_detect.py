import sys, numpy as np
from PIL import Image
from feature_extractor import get_model, get_features
from train import load_keras_model
from sklearn.preprocessing import normalize
from sklearn.neighbors import KDTree

WORD2VEC = "../data/class_vectors.npy"
MODEL_DIR = "../model/"

img_path = sys.argv[1]
img = Image.open(img_path).convert("RGB").resize((224,224))

# feature extractor
vgg = get_model()
feat = get_features(vgg, img)
print("feature shape:", feat.shape)

feat = normalize(feat, norm="l2")

# load ZSL model
zsl = load_keras_model(MODEL_DIR)
pred = zsl.predict(feat)          # (1,300)
print("pred shape:", pred.shape)

classes, vecs = zip(*sorted(np.load(WORD2VEC, allow_pickle=True), key=lambda x:x[0]))
tree = KDTree(np.asarray(vecs, float))
dist, idx = tree.query(pred, k=5)

for d,i in zip(dist[0], idx[0]):
    print(f"{classes[i]:15s}  {d:.4f}")
