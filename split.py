import argparse
import pandas as pd
from visualize import plot_feature_histograms
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from utils import open_file


fraction = 0.2
auc_threshold = 0.55
corr_threshold = 0.9


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
    df = open_file(file_path, header_in_file=False)
    #print("INFO: ", df.info())
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df
    #df.hist(figsize=(12, 10), bins=20)
    #plt.tight_layout()
    #plt.show()


def reduce_noise(test, df):
    features = [c for c in df.columns if c not in ["id", "diagnosis"]]
    y = df['diagnosis']
    aucs = {}
    to_remove = []
    # Remove features which have low AUC meaning they don't difenciate a lot between diagnosis
    for feature in features:
        auc = roc_auc_score(y, df[feature])
        if auc <= auc_threshold:
            df.drop(feature, axis=1, inplace=True)
            test.drop(feature, axis=1, inplace=True)
            features.remove(feature)
            print(f'removing AUC for {feature} is {auc}')
        aucs[feature] = auc

    rf = RandomForestClassifier(random_state=42)
    X_train = df.drop(columns=['diagnosis'])
    rf.fit(X_train, y)
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    print(feature_importance_df)
    features_to_remove = feature_importance_df[feature_importance_df['importance'] <= 0.01]['feature']
    df.drop(columns=features_to_remove, axis=1, inplace=True)
    test.drop(columns=features_to_remove, axis=1, inplace=True)

    return test, df
    # Remove features which which overlap a lot using correlation matrix
    # matrix = df.corr()
    # overlaped = {}
    # for i, feature_1 in enumerate(features):
    #     for j, feature_2 in enumerate(features):
    #         if i < j:
    #             corr_value = abs(matrix.loc[feature_1, feature_2])
    #             if corr_value >= corr_threshold:
    #                 overlaped[feature_1, feature_2] = corr_value
    #                 print(f'adding to overlaped {feature_1} and {feature_2}. VALUE: {corr_value}')
    #         else:
    #             continue
    
    # for key, value in overlaped.items():
    #     #print("AAA: ", key[0])
    #     if aucs[key[0]] < aucs[key[1]]: # lacks checking if the feature is in another pair as well
    #         df = df.drop(columns=[key[0]])
    #     else:
    #         df = df.drop(columns=[key[1]])


def normalize(test, train):
    train_features = train.drop(columns=['diagnosis'])
    test_features = test.drop(columns=['diagnosis'])
    mean = train_features.mean()
    std = train_features.std()
    train_norm = (train_features - mean) / std
    test_norm = (test_features - mean) / std

    train_norm['diagnosis'] = train['diagnosis'].values
    test_norm['diagnosis'] = test['diagnosis'].values

    return test_norm, train_norm


def main():
    parser = argparse.ArgumentParser(description='EDA')
    parser.add_argument('dataset')
    parser.add_argument('--plots', action='store_true')
    args = parser.parse_args()

    df = parse(args.dataset)
    if args.plots:
        plot_feature_histograms(df)
    
    test, train = split_dataset(df)
    test, train = reduce_noise(test, train)
    test, train = normalize(test, train)
    test.to_csv("test.csv", index=False)
    train.to_csv("train.csv", index=False)


if __name__ == "__main__":
    main()
