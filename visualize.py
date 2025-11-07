import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3D plotting
import numpy as np


def plot_loss_accuracy(dic, epoch):
    """Shows 2 learning curve graphs displayed at the end of the training."""
    loss = dic['loss']
    val_loss = dic['val_loss']
    acc = dic['acc']
    val_acc = dic['val_acc']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
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


def plot_precision_recall(dic, epoch):
    """Shows If validation precision ↑ and recall ↓, your model is getting too conservative (underfitting positives).
    If both go down after some epochs, you’re overfitting.
    If F1 stops improving, trigger early stopping there instead of only relying on loss.."""
    precision = dic['precision']
    val_precision = dic['val_precision']
    recall = dic['recall']
    val_recall = dic['val_recall']
    f1 = dic['f1']
    val_f1 = dic['val_f1']

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))
    # --- Precision ---
    ax1.plot(precision)
    ax1.plot(val_precision)
    if epoch is not None:
        ax1.axvline(x = epoch, color = 'r', linestyle = '--')
    ax1.set_title("Precision", fontsize=22)
    ax1.legend(["training precision", "validation precision", "early stop"])
    ax1.set_xlabel('epochs', fontsize=14)
    ax1.set_ylabel('loss', fontsize=14)
    ax1.tick_params(axis='both', labelsize=12)
    
    # --- Recall ---
    ax2.plot(recall)
    ax2.plot(val_recall)
    if epoch is not None:
        ax2.axvline(x = epoch, color = 'r', linestyle = '--')
    ax2.set_title("Recall", fontsize=22)
    ax2.legend(["training recall", "validation recall", "early stop"])
    ax2.set_xlabel('epochs', fontsize=14)
    ax2.set_ylabel('accuracy', fontsize=14)
    ax2.tick_params(axis='both', labelsize=12)

    # --- F1 ---
    ax3.plot(f1)
    ax3.plot(val_f1)
    if epoch is not None:
        ax3.axvline(x = epoch, color = 'r', linestyle = '--')
    ax3.set_title("f1", fontsize=22)
    ax3.legend(["training f1", "validation f1", "early stop"])
    ax3.set_xlabel('epochs', fontsize=14)
    ax3.set_ylabel('accuracy', fontsize=14)
    ax3.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig("plot_precision_recall.png")


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
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))
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


# wireframe plotting
# Train  MLP once normally using Adam.
# This gives final trained model with weights.
# Pick 2 weight parameters (for example, W[0][0] and W[1][0]).
# Create small variations around those weights:
# Slightly increase/decrease them in small steps.
# For each (theta 0, theta 1) combination, replace the weights in model temporarily.
# Compute the loss for the training (or validation) set with these modified weights.
# This gives the Z value for that (X,Y) pair.
# Plot those (theta 0 , theta 1, loss) values as a wireframe.

def plot_wireframe(X, Y, Z, title="Wireframe Plot", data_set):

    """Plots a 3D wireframe given X, Y, Z coordinates."""
    data = np.load(data_set, allow_pickle=True)
    params = {key: data[key] for key in data.files if key != 'topology'}
    w0 = params[f"W1"][0, 0]
    w1 = params[f"W1"][1, 0]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)
    ax.set_title(title)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.tight_layout()
    plt.savefig("wireframe_plot.png")