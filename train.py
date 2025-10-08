import numpy as np
import argparse
import pickle
import pandas as pd
from data_pipeline import Preprocessor
from utils import open_file, save_params, accuracy_score, precision_score, recall_score, f1_score
from visualize import plot_loss_accuracy
from base import BaseNetwork


class NeuronalNetwork(BaseNetwork):
    def __init__(self, train_set, test_set, layer, epochs, loss, batch_size):
        super().__init__()
        self.df = train_set
        self.test_set = test_set
        Y = self.test_set.iloc[:, 0].values
        self.test_Y = np.eye(2)[Y.astype(int)]
        self.configure(self.df.iloc[:, 1:].shape[1], self.test_Y, layer, output_size=2)
        self.X = self.df.iloc[:, 1:].values
        y = self.df.iloc[:, 0].values
        self.Y = np.eye(2)[y.astype(int)]
        self.epochs = epochs
        self.loss = loss
        self.batch_size = batch_size
        self.cache = {}
        self.first = True
        self.early_stop = False
        self.stop_epoch = None
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
        m = self.X.shape[0]
        best_val_loss = float('inf')
        patience = 20
        wait = 0

        for epoch in range(1, self.epochs + 1):
            idx = np.random.permutation(m)
            X_shuffled, Y_shuffled = self.X[idx], self.Y[idx]
            epoch_loss = 0
            all_y_true = []
            all_y_pred = []

            for i in range(0, m, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                Y_batch = Y_shuffled[i:i+self.batch_size]
                self.cache = {'A0': X_batch}
                
                for l in range(1, len(self.layer_sizes)):
                    A_prev = self.cache[f'A{l - 1}']
                    prediction = self.layers[l - 1].forward(l, A_prev)
                    self.cache.update(prediction)

                all_y_true.extend(Y_batch)
                all_y_pred.extend(self.cache[f'A{len(self.layer_sizes) - 1}'])
                loss, dA = self.layers[l - 1].categoricalCrossentropy(Y_batch, self.cache[f'A{l}'])
                epoch_loss += loss
                
                for l in reversed(range(l, len(self.layer_sizes))):
                    dA_prev, dW, dB = self.layers[l - 1].backward(dA, self.cache, l)
                    dA = dA_prev
                    if optimization == "sgd":
                        self.layers[l - 1].weights -= learning_rate * dW
                        self.layers[l - 1].biases -= learning_rate * dB
                    else:
                        self.layers[l - 1].adam_optimization(dW, dB, epoch, learning_rate)

            histor_rounded = {k : np.round(v, 4) for k, v in self.history.items()}
            pd.DataFrame(histor_rounded).to_csv("metrics_history.csv", index=False)

            epoch_loss /= m // self.batch_size
            val_pred, val_loss = self.forward_only(self.test_set, self.layers[l - 1].categoricalCrossentropy)
            self.save_metrics(epoch_loss, val_loss, val_pred, all_y_true, all_y_pred)
            if (epoch % 10 == 0 or epoch == 1) and self.early_stop == False:
                log(epoch, epoch_loss, val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience and self.early_stop == False:
                    self.stop_epoch = epoch
                    print(f"\nEarly stopping at {epoch}")
                    self.early_stop = True
                    save_params(self)


def main():
    parser = argparse.ArgumentParser(description='Predicts the cancer based on dataset')
    parser.add_argument('train_set', type=str, help="Path to training dataset (CSV)")
    parser.add_argument('test_set', type=str, help="Path to test dataset (CSV)")
    parser.add_argument('--layer', nargs="+", type=int, help="Hidden layer sizes (e.g. --layer 10 10)")
    parser.add_argument('--epochs', type=int, help="Number of training epochs")
    parser.add_argument('--loss', type=str, help="Loss function")
    parser.add_argument('--batch_size', type=int, default=32, help="Mini-batch size")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--optimizer', type=str, choices=["adam", "sgd"], default="adam", help="Optimizer to use (default: adam, options: adam, sgd)")
    args = parser.parse_args()

    pre = Preprocessor()
    train_df = open_file(args.train_set)
    test_df = open_file(args.test_set)

    train_df = pre.fit(train_df)
    test_df = pre.transform(test_df)

    # Save preprocessor for later prediction
    with open("preprocessor.pkl", "wb") as f:
        pickle.dump(pre, f)

    model = NeuronalNetwork(train_df, test_df, args.layer, args.epochs, args.loss, args.batch_size)
    model.create_layers()
    model.train(model, args.learning_rate, args.optimizer)
    plot_loss_accuracy(model.history, model.stop_epoch)
    if model.early_stop == False:
        save_params(model)


if __name__ == "__main__":
    main()
