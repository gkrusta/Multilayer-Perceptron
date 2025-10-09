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


def open_file(file_path, replace_label=True):
    """Reads a CSV file. removes first index column, puts numerical indexes to classes
    and returns it as a pandas DataFrame (with or without header)."""
    try:
        data = pd.read_csv(file_path, names=columns)
        if replace_label:
            data = data.drop(data.columns[0], axis=1)
            data.iloc[:, 0] = data.iloc[:, 0].map({'M': 1, 'B': 0})
        return data
    except Exception as e:
        print(e)
        exit(1)


def save_params(model):
    """Saves all model layer weights, biases, and topology into 'model.npz'."""
    params = {}
    params["topology"] = np.array(model.layer_sizes)

    for i, layer in enumerate(model.layers, start=1):
        params[f"W{i}"] = layer.weights
        params[f"B{i}"] =  layer.biases

    np.savez("model.npz", **params)


def relu(x):
    """Returns x if positive, otherwise 0."""
    return x * (x > 0)


def relu_backward(x):
    """1 for positive inputs, 0 otherwise."""
    return (x > 0).astype(float)


def softmax(x):
    """Converts logits to probabilities that sum to 1 across each row."""
    exp_logits = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def sigmoid(x):
    """Squashes values into range (0, 1)."""
    return 1 / (1 + np.exp(-x))


def accuracy_score(y_true, y_pred):
    """Percentage of correctly predicted samples."""
    return np.sum(y_true == y_pred) / len(y_true)


def precision_score(y_true, y_pred):
    """Of all predicted positives how many were actually positive."""
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    if TP + FP == 0:
        return 0.0
    return TP / (TP + FP)


def recall_score(y_true, y_pred):
    """Of all actual positives, how many were correctly predicted."""
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    if TP + FN == 0:
        return 0.0
    return TP / (TP + FN)


def f1_score(y_true, y_pred):
    """Harmonic mean of precision and recall."""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
