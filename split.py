import argparse
import pandas as pd
from visualize import plot_feature_histograms
from utils import open_file


fraction = 0.2


def split_dataset(df):
    """Split dataset into test and traininf sets which have equal B and M distibution."""
    grouped = df.groupby(df.diagnosis)
    b_grouped = grouped.get_group("B")
    m_grouped = grouped.get_group("M")
    b_test = b_grouped.sample(frac=fraction, random_state=42)
    m_test = m_grouped.sample(frac=fraction, random_state=42)
    b_train = b_grouped.drop(b_test.index)
    m_train = m_grouped.drop(m_test.index)

    test = pd.concat([b_test, m_test]).sample(frac=1)
    train = pd.concat([b_train, m_train]).sample(frac=1)
    return test, train


def main():
    parser = argparse.ArgumentParser(description='EDA')
    parser.add_argument('dataset')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()

    df = open_file(args.dataset, replace_label=False)
    if args.plots:
        plot_feature_histograms(df)
    
    test, train = split_dataset(df)
    test.to_csv("test.csv", index=False, header=False)
    train.to_csv("train.csv", index=False, header=False)


if __name__ == "__main__":
    main()
