import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from visualize import plot_feature_histograms
from sklearn.metrics import roc_auc_score


fraction = 0.2
auc_threshold = 0.55
corr_threshold = 0.9
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


def reduce_noise(df):
    features = [c for c in df.columns if c not in ["id", "diagnosis"]]
    y = df['diagnosis']
    aucs = {}

    # Remove features which have low AUC meaning they don't difenciate a lot between diagnosis
    for feature in features:
        auc = roc_auc_score(y, df[feature])
        if auc <= auc_threshold:
            df.drop(feature, axis=1, inplace=True)
            features.remove(feature)
            print(f'removing AUC for {feature} is {auc}')
        aucs[feature] = auc

    # Remove features which which overlap a lot using correlation matrix
    matrix = df.corr()
    overlaped = {}
    for i, feature_1 in enumerate(features):
        for j, feature_2 in enumerate(features):
            if i < j:
                corr_value = abs(matrix.loc[feature_1, feature_2])
                if corr_value >= corr_threshold:
                    overlaped[feature_1, feature_2] = corr_value
                    print(f'adding to overlaped {feature_1} and {feature_2}. VALUE: {corr_value}')
            else:
                continue
    
    for key, value in overlaped.items():
        #print("AAA: ", key[0])
        if aucs[key[0]] < aucs[key[1]]: # lacks checking if the feature is in another pair as well
            df = df.drop(columns=[key[0]])
        else:
            df = df.drop(columns=[key[1]])


def main():
    parser = argparse.ArgumentParser(description='EDA')
    parser.add_argument('dataset')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()

    df = parse(args.dataset)
    if args.plots:
        plot_feature_histograms(df)
    
    reduce_noise(df)
    # test, train = split_dataset(df)
    # test.to_csv("test.csv", index=False)
    # train.to_csv("train.csv", index=False)
    #if (len(sys.argv) < 2):
    #    print("Usage: python3 ./split.py dataset flag(optional)") 


if __name__ == "__main__":
    main()
