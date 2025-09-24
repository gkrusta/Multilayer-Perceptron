import matplotlib.pyplot as plt


def plot_loss(loss, val_loss, acc, val_acc):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,16))
    ax1.plot(loss)
    ax1.plot(val_loss)
    ax1.set_title("Loss", fontsize=22)
    ax1.legend(["training loss", "validation loss"])
    ax1.set_xlabel('epochs', fontsize=14)
    ax1.set_ylabel('loss', fontsize=14)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.grid(linestyle='-.')

    ax2.plot(acc)
    ax2.plot(val_acc)
    ax2.set_title("Accuracy", fontsize=22)
    ax2.legend(["training acc", "validation acc"])
    ax2.set_xlabel('epochs', fontsize=14)
    ax2.set_ylabel('accuracy', fontsize=14)
    ax2.tick_params(axis='both', labelsize=12)
    plt.show()


def diagnosis_bar(df):
    counts = df["diagnosis"].value_counts().sort_index()
    plt.figure(figsize=(10, 10))
    plt.bar(counts.index, counts.values)
    for i, val in enumerate(counts.values):
        plt.text(counts.index[i], val + 4, str(val))
    plt.xlabel("diagnosis (B, M)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()


def plot_feature_histograms(df):
    diagnosis_bar(df)

    features = [c for c in df.columns if c not in ["id", "diagnosis"]]
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
