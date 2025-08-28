import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def plot_feature_histograms(df):
    #print("INFO: ", df.info())
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
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
    index = 0
    for i in range(rows):
        for j in range(cols):
            feature = features[index]
            ax = axes[i][j]
            ax.hist(df[df['diagnosis']==0][feature], alpha=0.5, label="B (0)")
            ax.hist(df[df['diagnosis']==1][feature], alpha=0.5, label="M (1)")
            ax.set_title(feature)
            index += 1
    plt.tight_layout()
    plt.show()
    #df.hist(figsize=(12, 10), bins=20)
    #plt.tight_layout()
    #plt.show()
