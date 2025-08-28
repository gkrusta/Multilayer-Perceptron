import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from visualize import plot_feature_histograms


fraction = 0.2

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


def split_dataset(df):
    grouped = df.groupby(df.diagnosis)
    b_grouped = grouped.get_group(0)
    m_grouped = grouped.get_group(1)

    b_test = b_grouped.sample(frac=fraction, random_state=42)
    m_test = m_grouped.sample(frac=fraction, random_state=42)
    b_train = b_grouped.drop(b_test.index)
    m_train = m_grouped.drop(m_test.index)

    test = pd.concat([b_test, m_test]).sample(frac=1)
    train = pd.concat([b_train, m_train]).sample(frac=1)
    
    return test, train


def parse(file_path):
    df = open_file(file_path)
    #print("INFO: ", df.info())
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df
    #df.hist(figsize=(12, 10), bins=20)
    #plt.tight_layout()
    #plt.show()


def main():
    parser = argparse.ArgumentParser(description='EDA')
    parser.add_argument('dataset')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()

    df = parse(args.dataset)
    if args.plots:
        plot_feature_histograms(df)
    
    test, train = split_dataset(df)
    test.to_csv("test.csv", index=False)
    train.to_csv("train.csv", index=False)
    #if (len(sys.argv) < 2):
    #    print("Usage: python3 ./split.py dataset flag(optional)") 


if __name__ == "__main__":
    main()
