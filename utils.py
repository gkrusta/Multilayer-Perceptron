import pandas as pd
import numpy as np


columns = [
    "id", "diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]


def open_file(file_path, header_in_file=True):
    try:
        if header_in_file:
            data = pd.read_csv(file_path)
        else:
            data = pd.read_csv(file_path, names=columns)
        return data
    except Exception as e:
        print(e)
        exit(1)


def save_params(model):
    params = {}
    params["topology"] = np.array(model.layer_sizes)

    for i, layer in enumerate(model.layers, start=1):
        params[f"W{i}"] = layer.weights
        params[f"B{i}"] =  layer.biases

    np.savez("model.npz", **params)


def relu(x):
    return x * (x > 0)


def relu_backward(x):
    return (x > 0).astype(float) 


def softmax(x):
    exp_logits = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    