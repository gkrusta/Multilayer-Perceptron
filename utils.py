import pandas as pd


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


def relu(x):
    return max(0.0, x)


def relu_der(x):
    if x > 0:
        return x
    else:
        return 0


    # def sigmoid(x):
    #     return (1 / (1 + np.exp(-x)))