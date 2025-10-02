import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


fraction = 0.01
auc_threshold = 0.55


class Preprocessor:
    def __init__(self):
        self.mean = None
        self.std = None
        self.features_to_keep = None


    def normalize(self, data_set):
        train_features = data_set.drop(columns=['diagnosis'])
        data_norm = (train_features - self.mean) / self.std
        data_norm['diagnosis'] = data_set['diagnosis'].values
        return data_norm


    def fit(self, train_set):
        features = [c for c in train_set.columns if c not in ["id", "diagnosis"]]
        y = train_set['diagnosis']
        aucs = {}

        # Remove features which have low AUC meaning they don't difenciate a lot between diagnosis
        for feature in features:
            auc = roc_auc_score(y, train_set[feature])
            if auc <= auc_threshold:
                train_set.drop(feature, axis=1, inplace=True)
                features.remove(feature)
                print(f'removing AUC for {feature} is {auc}')
            aucs[feature] = auc # remove

        rf = RandomForestClassifier(random_state=42)
        X_train = train_set.drop(columns=['diagnosis'])
        rf.fit(X_train, y)
        importances = rf.feature_importances_

        feature_importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        print(feature_importance_df)
        features_to_remove = feature_importance_df[feature_importance_df['importance'] <= fraction]['feature']
        train_set.drop(columns=features_to_remove, axis=1, inplace=True)
        
        self.features_to_keep = [c for c in train_set.columns if c != 'diagnosis']

        self.mean = train_set[self.features_to_keep].mean()
        self.std = train_set[self.features_to_keep].std()
        train_set = self.normalize(train_set)
        return train_set


    def transform(self, data_set):
        print(self.features_to_keep)
        cols = self.features_to_keep + ['diagnosis']
        df = data_set[cols]
        df = self.normalize(df)
        return df
