import numpy as np
import argparse
import pickle
import pandas as pd
from data_pipeline import Preprocessor
from utils import open_file, save_params, accuracy_score, precision_score, recall_score, f1_score
from visualize import plot_loss_accuracy, plot_precision_recall
from base import BaseNetwork

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3D plotting
from sklearn.decomposition import PCA
import imageio


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

    def _flatten_params(self):
        theta_list = []
        shapes = []
        for layer in self.layers:
            shapes.append(layer.weights.shape)
            theta_list.append(layer.weights.flatten())
            shapes.append(layer.biases.shape)
            theta_list.append(layer.biases.flatten())
        return np.concatenate(theta_list), shapes


    def _unflatten_params(self, flat, shapes):
        idx = 0
        ptr = 0
        for layer in self.layers:
            w_shape = shapes[ptr]; ptr += 1
            size = np.prod(w_shape)
            layer.weights = flat[idx:idx+size].reshape(w_shape)
            idx += size

            b_shape = shapes[ptr]; ptr += 1
            size = np.prod(b_shape)
            layer.biases = flat[idx:idx+size].reshape(b_shape)
            idx += size

    def train(self, log, learning_rate, optimization="adam"):
        """Main training loop with mini-batch gradient descent,
        Adam/SGD optimization, and early stopping."""
        m = self.X.shape[0]
        best_val_loss = float('inf')
        patience = 20
        wait = 0
        self.theta_history = []   # before training

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
                    theta, _ = self._flatten_params()
                    self.theta_history.append(theta.copy())

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
                    print(f"Early stopping at {epoch}")
                    self.early_stop = True
                    self.stop_epoch = epoch
                    save_params(self)


    def plot_true_loss_landscape(self, span=10.0, n=61):
        """
        2D Loss Landscape + Optimizer Path (aligned):
        - PCA from actual optimizer trajectory
        - Surface heatmap
        - Optimizer path ON the surface (black line)
        """

        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        # -----------------------------------------------------------
        # 1. Flatten original parameters
        # -----------------------------------------------------------
        theta_orig, shapes = self._flatten_params()
        d = theta_orig.size
        print(f"[INFO] Total parameters: {d}")

        theta_history = np.array(self.theta_history)
        print(f"[INFO] Optimizer steps recorded: {theta_history.shape[0]}")

        # Compute Δθ for PCA
        delta = theta_history - theta_history[0]   # (steps, d)

        # -----------------------------------------------------------
        # 2. PCA directions from training trajectory
        # -----------------------------------------------------------
        pca = PCA(n_components=2)
        pca.fit(delta)

        U = pca.components_[0]
        V = pca.components_[1]

        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)

        # -----------------------------------------------------------
        # 3. Project training path into α–β coordinates
        # -----------------------------------------------------------
        alpha_path = delta @ U
        beta_path  = delta @ V

        # Stretch the path to fill the span visually
        path_scale = span / (np.max(np.abs(alpha_path)) + 1e-8)
        alpha_path *= path_scale
        beta_path  *= path_scale

        # -----------------------------------------------------------
        # 4. Build the PCA grid
        # -----------------------------------------------------------
        alphas = np.linspace(-span, span, n)
        betas  = np.linspace(-span, span, n)
        A, B = np.meshgrid(alphas, betas)
        Z = np.zeros_like(A)

        # -----------------------------------------------------------
        # 5. Sweep the loss landscape: compute Z for each grid point
        # -----------------------------------------------------------
        print("[INFO] Computing loss surface…")

        for i in range(n):
            for j in range(n):
                θ_new = theta_orig + A[i, j] * U + B[i, j] * V
                self._unflatten_params(θ_new, shapes.copy())

                _, loss = self.forward_only(self.test_set, self.layers[-1].categorical_cross_entropy)
                Z[i, j] = loss

        # Restore the original weights
        self._unflatten_params(theta_orig, shapes.copy())

        # Normalize Z for visual contrast
        Z = Z - Z.min()

        # -----------------------------------------------------------
        # 6. Compute Z positions for optimizer path (so line lies ON surface)
        # -----------------------------------------------------------
        Z_path = []
        for a, b in zip(alpha_path, beta_path):
            i = np.argmin(np.abs(alphas - a))
            j = np.argmin(np.abs(betas - b))
            Z_path.append(Z[i, j])

        Z_path = np.array(Z_path) + 0.02  # Slight lift above surface

        # -----------------------------------------------------------
        # 7. Plot: surface + optimizer path
        # -----------------------------------------------------------
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
        plt.show()

        print("[INFO] Landscape + optimizer path generated successfully.")


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
    model.plot_true_loss_landscape()


if __name__ == "__main__":
    main()
