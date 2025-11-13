import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3D plotting
import numpy as np

plt.style.use("seaborn-v0_8-darkgrid")   # modern clean style
plt.rcParams["figure.dpi"] = 140          # crisp resolution
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
def plot_loss_accuracy(dic, epoch):
    loss = dic['loss']
    val_loss = dic['val_loss']
    acc = dic['acc']
    val_acc = dic['val_acc']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # --- Loss ---
    ax1.plot(loss, color="#1f77b4", linewidth=2, label="Training Loss")
    ax1.plot(val_loss, color="#ff7f0e", linewidth=2, label="Validation Loss")

    if epoch is not None:
        ax1.axvline(epoch, color="red", linestyle="--", linewidth=2, label="Early Stop")

    ax1.set_title("Training vs Validation Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # --- Accuracy ---
    ax2.plot(acc, color="#2ca02c", linewidth=2, label="Training Accuracy")
    ax2.plot(val_acc, color="#d62728", linewidth=2, label="Validation Accuracy")

    if epoch is not None:
        ax2.axvline(epoch, color="red", linestyle="--", linewidth=2)

    ax2.set_title("Training vs Validation Accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle(
        "Neural Network Training Curves\n(Data explored earlier using Random Forest & AUC score)",
        fontsize=16
    )
    plt.tight_layout()
    plt.savefig("plot_loss_accuracy.png", bbox_inches="tight")

def plot_precision_recall(dic, epoch):
    precision = dic['precision']
    val_precision = dic['val_precision']
    recall = dic['recall']
    val_recall = dic['val_recall']
    f1 = dic['f1']
    val_f1 = dic['val_f1']

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    titles = ["Precision", "Recall", "F1 Score"]
    colors = [["#1f77b4", "#ff7f0e"], ["#2ca02c", "#d62728"], ["#9467bd", "#8c564b"]]
    data = [
        (precision, val_precision),
        (recall, val_recall),
        (f1, val_f1),
    ]

    for ax, title, (train, val), (c1, c2) in zip(axes, titles, data, colors):
        ax.plot(train, color=c1, linewidth=2, label="Train")
        ax.plot(val, color=c2, linewidth=2, label="Validation")

        if epoch is not None:
            ax.axvline(epoch, color="red", linestyle="--", linewidth=2, label="Early Stop")

        ax.set_title(title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle(
        "Precision, Recall, and F1 Trends\n",
        fontsize=16
    )
    plt.tight_layout()
    plt.savefig("plot_precision_recall.png", bbox_inches="tight")

def diagnosis_bar(df):
    counts = df["diagnosis"].value_counts().sort_index()
    plt.figure(figsize=(8, 7))

    bars = plt.bar(
        counts.index,
        counts.values,
        color=["#1f77b4", "#d62728"],
        edgecolor="black"
    )

    for bar in bars:
        y = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            y + max(counts.values)*0.02,
            str(int(y)),
            ha="center",
            fontsize=14,
            fontweight="bold",
            color="black"
        )

    plt.title("Diagnosis Distribution\n(B = Benign, M = Malignant)", fontsize=18)
    plt.xlabel("Diagnosis")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("diagnosis_bar.png", bbox_inches="tight")


def plot_feature_histograms(df):
    diagnosis_bar(df)

    features = [c for c in df.columns if c not in ["id", "diagnosis"]]
    rows, cols = 5, 6
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
    index = 0

    for i in range(rows):
        for j in range(cols):
            ax = axes[i][j]
            feature = features[index]

            ax.hist(df[df["diagnosis"] == "B"][feature], bins=20, alpha=0.5, color="#1f77b4")
            ax.hist(df[df["diagnosis"] == "M"][feature], bins=20, alpha=0.5, color="#d62728")

            ax.set_title(feature, fontsize=10)
            ax.tick_params(axis='both', labelsize=8)
            ax.set_yticks([])
            index += 1

    fig.legend(["Benign", "Malignant"], loc="lower center", fontsize=14)
    plt.suptitle(
        "Feature Distributions by Diagnosis",
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig("feature_histograms.png", bbox_inches="tight")

