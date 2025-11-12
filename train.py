import numpy as np
import argparse
import pickle
import pandas as pd
from data_pipeline import Preprocessor
from utils import open_file, save_params, accuracy_score, precision_score, recall_score, f1_score
from visualize import plot_loss_accuracy, plot_precision_recall
from base import BaseNetwork


class NeuronalNetwork(BaseNetwork):
    """Implements the full multi layer perceptron training pipeline:
    - Data preparation
    - Forward + backward propagation
    - Metrics and early stopping"""
    def __init__(self, train_set, test_set, layer, epochs, loss, batch_size):
        super().__init__()
        self.df = train_set
        self.test_set = test_set # prepare test labels (one hot encoded)
        Y = self.test_set.iloc[:, 0].values
        self.test_Y = np.eye(2)[Y.astype(int)]

        self.X = self.df.iloc[:, 1:].values # prepare training data and labels
        y = self.df.iloc[:, 0].values
        self.Y = np.eye(2)[y.astype(int)]

        self.configure(self.df.iloc[:, 1:].shape[1], self.test_Y, layer, output_size=2)
        self.epochs = epochs
        if loss != 'categorical_cross_entropy':
            raise ValueError(f"Unknown loss: {loss}")
        self.batch_size = batch_size
        self.cache = {}
        self.first = True
        self.early_stop = False
        self.stop_epoch = None

        # history for visualization
        self.history = {
            "loss": [],
            "val_loss": [],
            "acc": [],
            "val_acc": [],
            "precision": [],
            "val_precision": [],
            "recall": [],
            "val_recall": [],
            "f1": [],
            "val_f1": []
        }


    def __call__(self, epoch, loss, val_loss):
        if self.first:
            print("\nx_train shape : ", self.df.shape)
            print("x_valid shape : ", self.test_set.shape)
            self.first = False
        print(f"epoch {epoch:02d}/{self.epochs} - loss: {loss:.4f} - val_loss: {val_loss:.4f}")


    def save_metrics(self, loss, val_loss, val_y_pred, all_y_true, all_y_pred) -> None:
        # Convert one hot encoded arrays to true class index
        y_true = np.argmax(all_y_true, axis=1)
        y_pred = np.argmax(all_y_pred, axis=1)
        y_val_true = np.argmax(self.test_Y, axis=1)
        y_val_pred = np.argmax(val_y_pred, axis=1)

        # --- Training metrics ---
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # --- Validation metrics ---
        val_acc = accuracy_score(y_val_true, y_val_pred)
        val_precision = precision_score(y_val_true, y_val_pred)
        val_recall = recall_score(y_val_true, y_val_pred)
        val_f1 = f1_score(y_val_true, y_val_pred)

        # --- Save to history ---
        self.history["loss"].append(loss)
        self.history["val_loss"].append(val_loss)
        self.history["acc"].append(acc)
        self.history["val_acc"].append(val_acc)
        self.history["precision"].append(precision)
        self.history["recall"].append(recall)
        self.history["f1"].append(f1)
        self.history["val_precision"].append(val_precision)
        self.history["val_recall"].append(val_recall)
        self.history["val_f1"].append(val_f1)


    def train(self, log, learning_rate, optimization="adam"):
        """Main training loop with mini-batch gradient descent,
        Adam/SGD optimization, and early stopping."""
        m = self.X.shape[0]
        best_val_loss = float('inf')
        patience = 20
        wait = 0

        for epoch in range(1, self.epochs + 1):
            idx = np.random.permutation(m) # shuffle dataset each epoch
            X_shuffled, Y_shuffled = self.X[idx], self.Y[idx]
            epoch_loss = 0
            all_y_true = []
            all_y_pred = []

            for i in range(0, m, self.batch_size): 
                X_batch = X_shuffled[i:i+self.batch_size]
                Y_batch = Y_shuffled[i:i+self.batch_size]
                self.cache = {'A0': X_batch}
                
                for l in range(1, len(self.layer_sizes)): # forward pass through all layers
                    A_prev = self.cache[f'A{l - 1}']
                    prediction = self.layers[l - 1].forward(l, A_prev)
                    self.cache.update(prediction)

                all_y_true.extend(Y_batch)
                all_y_pred.extend(self.cache[f'A{len(self.layer_sizes) - 1}'])
                loss, dA = self.layers[l - 1].categorical_cross_entropy(Y_batch, self.cache[f'A{l}']) # compute loss + first gradient
                epoch_loss += loss

                for l in reversed(range(1, len(self.layer_sizes))): # backward pass (through all layers)
                    dA_prev, dW, dB = self.layers[l - 1].backward(dA, self.cache, l)
                    dA = dA_prev
                    if optimization == "sgd":
                        self.layers[l - 1].weights -= learning_rate * dW
                        self.layers[l - 1].biases -= learning_rate * dB
                    else:
                        self.layers[l - 1].adam_optimization(dW, dB, epoch, learning_rate)

            histor_rounded = {k : np.round(v, 4) for k, v in self.history.items()} # save metrics to CSV each epoch
            pd.DataFrame(histor_rounded).to_csv("metrics_history.csv", index=False)

            epoch_loss /= m // self.batch_size # compute average loss + validation metrics
            val_pred, val_loss = self.forward_only(self.test_set, self.layers[-1].categorical_cross_entropy)
            self.save_metrics(epoch_loss, val_loss, val_pred, all_y_true, all_y_pred)
            if (epoch % 10 == 0 or epoch == 1) and self.early_stop == False:  # logging every 10 epochs
                log(epoch, epoch_loss, val_loss)
            if val_loss < best_val_loss: #early stopping
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience and self.early_stop == False:
                    self.stop_epoch = epoch
                    print(f"Early stopping at {epoch}")
                    self.early_stop = True
                    save_params(self)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3D plotting

def plot_wireframe(self, model):
    # --- 1) Load saved weights ---
    params = np.load("model.npz", allow_pickle=True)
    # Example if you saved with keys W1, B1, W2, B2, ...
    # Build a dict
    loaded = {k: params[k] for k in params.files}

    for i, layer in enumerate(model.layers, start=1):
        layer.weights = loaded[f"W{i}"].copy()
        layer.biases  = loaded[f"B{i}"].copy()
        
    # --- 3) Pick two weight entries from one layer as axes ---
    L = 2  # pick layer index in [1..num_layers], choose a hidden or output layer
    W = model.layers[L-1].weights  # shape (out_dim, in_dim)
    # pick two random coordinates in this W
    rng = np.random.default_rng(42)
    i1 = rng.integers(0, W.shape[0])
    j1 = rng.integers(0, W.shape[1])
    i2 = rng.integers(0, W.shape[0])
    j2 = rng.integers(0, W.shape[1])

    w1_orig = W[i1, j1].copy()
    w2_orig = W[i2, j2].copy()

    # --- 4) Build a grid of small perturbations around these two weights ---
    n = 31
    span1 = 0.25  # how far to move w1 (tune)
    span2 = 0.25  # how far to move w2 (tune)
    d1 = np.linspace(-span1, span1, n)
    d2 = np.linspace(-span2, span2, n)
    D1, D2 = np.meshgrid(d1, d2)

    loss_surface = np.zeros_like(D1)


    # --- 5) Sweep the grid: set weights → forward → loss ---
    for a in range(n):
        for b in range(n):
            # set perturbed weights
            model.layers[L-1].weights[i1, j1] = w1_orig + D1[a, b]
            model.layers[L-1].weights[i2, j2] = w2_orig + D2[a, b]

            # compute loss
            loss_surface[a, b] = self.forward_only(self.test_set, self.layers[-1].categorical_cross_entropy)

    # restore original weights (important!)
    model.layers[L-1].weights[i1, j1] = w1_orig
    model.layers[L-1].weights[i2, j2] = w2_orig

    # --- 6) Plot wireframe: X=Δw1, Y=Δw2, Z=loss ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(D1, D2, loss_surface)  # keep default colors per your policy/tooling
    ax.set_xlabel(f"ΔW[L={L}][{i1},{j1}]")
    ax.set_ylabel(f"ΔW[L={L}][{i2},{j2}]")
    ax.set_zlabel("Loss (categorical CE)")
    ax.set_title("Loss Landscape around Trained Weights")
    plt.show()


def main():
    """Entry point for training the MLP model from the command line."""
    parser = argparse.ArgumentParser(description='Predicts the cancer based on dataset')
    parser.add_argument('train_set', type=str, help="Path to training dataset (CSV)")
    parser.add_argument('test_set', type=str, help="Path to test dataset (CSV)")
    parser.add_argument('--layer', nargs="+", default=[12, 12], type=int, help="Hidden layer sizes (e.g. --layer 10 10)")
    parser.add_argument('--epochs', type=int, help="Number of training epochs")
    parser.add_argument('--loss', type=str, help="Loss function")
    parser.add_argument('--batch_size', type=int, default=16, help="Mini-batch size")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--optimizer', type=str, choices=["adam", "sgd"], default="adam", help="Optimizer to use (default: adam, options: adam, sgd)")
    args = parser.parse_args()

    pre = Preprocessor()
    train_df = open_file(args.train_set)

    test_df = open_file(args.test_set)
    train_df = pre.fit(train_df)
    test_df = pre.transform(test_df)

    with open("preprocessor.pkl", "wb") as f: # save preprocessor for later prediction to use on test set
        pickle.dump(pre, f)

    model = NeuronalNetwork(train_df, test_df, args.layer, args.epochs, args.loss, args.batch_size)
    model.create_layers()
    model.train(model, args.learning_rate, args.optimizer)
    plot_loss_accuracy(model.history, model.stop_epoch)
    plot_precision_recall(model.history, model.stop_epoch)
    if model.early_stop == False:
        save_params(model)
    plot_wireframe(model)


if __name__ == "__main__":
    main()
