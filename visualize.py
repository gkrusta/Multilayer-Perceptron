import matplotlib.pyplot as plt


def plot_loss_accuracy(dic, epoch):
    """Shows 2 learning curve graphs displayed at the end of the training."""
    loss = dic['loss']
    val_loss = dic['val_loss']
    acc = dic['acc']
    val_acc = dic['val_acc']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 14))
    # --- Loss ---
    ax1.plot(loss)
    ax1.plot(val_loss)
    if epoch is not None:
        ax1.axvline(x = epoch, color = 'r', linestyle = '--')
    ax1.set_title("Loss", fontsize=22)
    ax1.legend(["training loss", "validation loss", "early stop"])
    ax1.set_xlabel('epochs', fontsize=14)
    ax1.set_ylabel('loss', fontsize=14)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.grid(linestyle='-.')
    
    # --- Accuracy ---
    ax2.plot(acc)
    ax2.plot(val_acc)
    if epoch is not None:
        ax2.axvline(x = epoch, color = 'r', linestyle = '--')
    ax2.set_title("Accuracy", fontsize=22)
    ax2.legend(["training acc", "validation acc", "early stop"])
    ax2.set_xlabel('epochs', fontsize=14)
    ax2.set_ylabel('accuracy', fontsize=14)
    ax2.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig("plot_loss_accuracy.png")


def diagnosis_bar(df):
    """Displays 2 bars showing which cases there are more- positive or negative."""
    counts = df["diagnosis"].value_counts().sort_index()
    plt.figure(figsize=(12, 12))
    plt.bar(counts.index, counts.values)
    for i, val in enumerate(counts.values):
        plt.text(counts.index[i], val + 4, str(val))
    plt.xlabel("diagnosis (B, M)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig("diagnosis_bar.png")


def plot_feature_histograms(df):
    """Displays how different the distribution of clases is between features."""
    diagnosis_bar(df)

    features = [c for c in df.columns if c not in ["id", "diagnosis"]]
    rows, cols = 5, 6
    fig, axes = plt.subplots(rows, cols, figsize=(18, 16))
    index = 0
    for i in range(rows):
        for j in range(cols):
            feature = features[index]
            ax = axes[i][j]
            ax.hist(df[df['diagnosis']=="B"][feature], alpha=0.5, label="B (0)")
            ax.hist(df[df['diagnosis']=="M"][feature], alpha=0.5, label="M (1)")
            ax.set_title(feature)
            index += 1
    
    plt.tight_layout()
    plt.savefig("feature_histograms.png")
