import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3D plotting
import numpy as np
from sklearn.decomposition import PCA


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


def _flatten_params(model):
    theta_list = []
    shapes = []
    for layer in model.layers:
        shapes.append(layer.weights.shape)
        theta_list.append(layer.weights.flatten())
        shapes.append(layer.biases.shape)
        theta_list.append(layer.biases.flatten())
    return np.concatenate(theta_list), shapes


def _unflatten_params(model, flat, shapes):
    idx = 0
    ptr = 0
    for layer in model.layers:
        w_shape = shapes[ptr]; ptr += 1
        size = np.prod(w_shape)
        layer.weights = flat[idx:idx+size].reshape(w_shape)
        idx += size

        b_shape = shapes[ptr]; ptr += 1
        size = np.prod(b_shape)
        layer.biases = flat[idx:idx+size].reshape(b_shape)
        idx += size


def plot_true_loss_landscape(model, span=10.0, n=61):
    """
    2D Loss Landscape + Optimizer Path (aligned):
    - PCA from actual optimizer trajectory
    - Surface heatmap
    - Optimizer path on the surface (black line)
    """

    # Flatten original parameters
    theta_orig, shapes = _flatten_params(model)
    d = theta_orig.size
    print(f"[INFO] Total parameters: {d}")

    theta_history = np.array(model.theta_history)
    print(f"[INFO] Optimizer steps recorded: {theta_history.shape[0]}")

    # Compute theta0 for PCA
    delta = theta_history - theta_history[0]   # (steps, d)
    
    # PCA directions from training trajectory
    pca = PCA(n_components=2)
    pca.fit(delta)

    U = pca.components_[0]
    V = pca.components_[1]

    U /= np.linalg.norm(U)
    V /= np.linalg.norm(V)

    #Project training path into alfa-beta coordinates
    alpha_path = delta @ U
    beta_path  = delta @ V

    # Stretch the path to fill the span visually
    path_scale = span / (np.max(np.abs(alpha_path)) + 1e-8)
    alpha_path *= path_scale
    beta_path  *= path_scale

    # PCA grid
    alphas = np.linspace(-span, span, n)
    betas  = np.linspace(-span, span, n)
    A, B = np.meshgrid(alphas, betas)
    Z = np.zeros_like(A)

    # compute Z for each grid point
    print("[INFO] Computing loss surface…")

    for i in range(n):
        for j in range(n):
            θ_new = theta_orig + A[i, j] * U + B[i, j] * V
            _unflatten_params(model, θ_new, shapes.copy())

            _, loss = model.forward_only(model.test_set, model.layers[-1].categorical_cross_entropy)
            Z[i, j] = loss

    # Restore the original weights
    _unflatten_params(model, theta_orig, shapes.copy())

    # Normalize
    Z = Z - Z.min()

    # Compute Z positions for optimizer path so line lies ON surface
    Z_path = []
    for a, b in zip(alpha_path, beta_path):
        i = np.argmin(np.abs(alphas - a))
        j = np.argmin(np.abs(betas - b))
        Z_path.append(Z[i, j])

    Z_path = np.array(Z_path) + 0.02  # Slight lift above surface

    # Plot: surface + optimizer path
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        A, B, Z,
        cmap="viridis",
        edgecolor='none',
        antialiased=True
    )
    ax.plot(
        alpha_path,
        beta_path,
        Z_path,
        color='black',
        linewidth=2,
        marker='o',
        markersize=3,
        zorder=999, 
        label="Optimizer Path"
    )

    ax.set_xlabel("α (PCA direction 1)")
    ax.set_ylabel("β (PCA direction 2)")
    ax.set_zlabel("Loss (normalized)")
    ax.set_title("Loss Landscape with Optimizer Path (Aligned with PCA)")
    ax.legend()

    fig.colorbar(surf, shrink=0.6)
    plt.savefig("landscape_loss.png", bbox_inches="tight")
