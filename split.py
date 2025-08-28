import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
from visualize import plot_feature_histograms


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



def parse(file_path, flag=0):
    df = open_file(file_path)
    #print("INFO: ", df.info())
    df = df.drop(columns=['id'])
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    if (flag==1):
        plot_feature_histograms(df)
    split_dataset(df)
    #df.hist(figsize=(12, 10), bins=20)
    #plt.tight_layout()
    #plt.show()


def main():
    parser = argparse.ArgumentParser(description='EDA')
    parser.add_argument('dataset')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()
    print('ARGS: ', args.dataset, args.plots)
    #if (len(sys.argv) < 2):
    #    print("Usage: python3 ./split.py dataset flag(optional)") 
    parse(args.dataset)


if __name__ == "__main__":
    main()