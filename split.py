import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score


columns = [
    "id", "diagnosis",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

def open_file(file_path):
    try:
        data = pd.read_csv(file_path, names=columns)
        return data
    except Exception as e:
        print(e)
        exit(1)


def parse(file_path):
    df = open_file(file_path)
    #print("INFO: ", df.info())
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    counts = df["diagnosis"].value_counts().sort_index()
    plt.bar(counts, counts.values)
    plt.xlabel("diagnosis (B, M)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()
    df = df.drop(columns=['id'])

    features = [c for c in df.columns if c not in ["id", "diagnosis"]]
    y = df['diagnosis']
    aucs = {}
    for feature in features:
        auc = roc_auc_score(y, df[feature])
        print(f'AUC for {feature} is {auc}')
        aucs[feature] = auc

    rows, cols = 5, 6
    fig, axes = plt.subplots(rows, cols)
    
    for i, feature in enumerate(features):
        ax = axes[i]
        ax.hist(df[df['diagnosis']==0][feature], alpha=0.5, label="B (0)")
        ax.hist(df[df['diagnosis']==1][feature], alpha=0.5, label="M (1)")
        ax.set_tile(feature)
    plt.tight_layout()
    plt.suptitle("Histograms of features by diagnosis")
    plt.show()
    #df.hist(figsize=(12, 10), bins=20)
    #plt.tight_layout()
    #plt.show()


def main():
    if (len(sys.argv) < 2):
        print("Usage: python3 ./split.py dataset") 
    else:
        parse(sys.argv[1])


if __name__ == "__main__":
    main()