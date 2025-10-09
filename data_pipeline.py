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
        """Ignore headers- replace them with numeric indexes, normalizes data."""
        data_set.columns = range(data_set.shape[1])
        features = data_set.iloc[:, 1:]
        diagnosis_col = data_set.iloc[:, 0]
        data_norm = (features - self.mean.values) / self.std.values
        data_norm.insert(0, 0, diagnosis_col.values)
        return data_norm


    def fit(self, train_set):
        """
        Removes features which won't be helpful durning training using 2 methods:
        - AUC tells how well a feature can separate 2 classes- malignant (1) vs benign (0).
        Anything bellow 0.55 is weak or random guessing.
        - Random Forest model finds which features are most useful for prediction.
        """
        features = [c for c in train_set.columns if c not in ["id", "diagnosis"]]
        y = pd.to_numeric(train_set['diagnosis'], errors='coerce')

        for feature in features:
            auc = roc_auc_score(y, train_set[feature])
            if auc <= auc_threshold:
                train_set.drop(feature, axis=1, inplace=True)
                features.remove(feature)
                print(f'removing AUC for {feature} is {auc}')

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
        """Makes the validation data set have the same column names as the training set obtained before
        and adds normalization."""
        print(self.features_to_keep)
        cols = ['diagnosis'] + self.features_to_keep
        df = data_set[cols]
        df = self.normalize(df)
        return df
