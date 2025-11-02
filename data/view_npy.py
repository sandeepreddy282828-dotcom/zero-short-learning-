import numpy as np

file_path = "../data/class_vectors.npy"

data = np.load(file_path, allow_pickle=True)

# If it's a list of tuples like (label, vector)
if isinstance(data[0], tuple):
    for label, vector in data:
        print("Class:", label)
        print("Vector (first 5 values):", vector[:5])
        print("---")
else:
    print(data)
