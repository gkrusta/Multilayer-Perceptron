import numpy as np
import argparse
from utils import open_file
from visualize import plot_loss_accuracy
from layer import Layer
from sklearn.metrics import accuracy_score


class NeuronalNetwork:
    def __init__(self, train_set, test_set, layer, epochs, loss, batch_size):
        self.df = open_file(train_set)
        self.test_set = open_file(test_set)
        self.numer_of_inputs = self.df.drop(columns=['diagnosis']).shape[1]
        self.hiden_layers = layer
        self.X = self.df.iloc[:, :-1].values
        y = self.df.iloc[:, -1].values
        self.Y = np.eye(2)[y.astype(int)]
        self.output_size = 2
        self.epochs = epochs
        self.loss = loss
        self.batch_size = batch_size
        self.cache = {}
        self.first = True
        self.history = {
            "loss": [],
            "val_loss": [],
            "acc": [],
            "val_acc": []
        }


    def __call__(self, epoch, loss, val_loss):
        if self.first:
            print("x_train shape : ", self.df.shape)
            print("x_valid shape : ", self.test_set.shape)
            self.first = False
        print(f"epoch {epoch:02d}/{self.epochs} - loss: {loss:.4f} - val_loss: {val_loss:.4f}")


    def save_metrics(self, loss, val_loss, val_y_pred, all_y_true, all_y_pred):
        y_true = np.argmax(all_y_true, axis=1)
        y_pred = np.argmax(all_y_pred, axis=1)
        y_val_true = np.argmax(self.test_Y, axis=1)
        y_val_pred = np.argmax(val_y_pred, axis=1)
        acc = accuracy_score(y_true, y_pred)
        val_acc = accuracy_score(y_val_true, y_val_pred)

        self.history["loss"].append(loss)
        self.history["val_loss"].append(val_loss)
        self.history["acc"].append(acc)
        self.history["val_acc"].append(val_acc)


    def create_layers(self, activation='relu', output_activation='softmax'):
        self.layer_sizes = [self.numer_of_inputs] + self.hiden_layers + [self.output_size]
        if len(self.hiden_layers) == 1:
            self.layer_sizes.insert(2, self.hiden_layers[0])

        self.layers = []
        for i in range(1, len(self.layer_sizes)):
            act = output_activation if i == len(self.layer_sizes) - 1 else activation
            layer = Layer(self.layer_sizes[i - 1], self.layer_sizes[i], act)
            self.layers.append(layer)


    def forward_only(self):
        X = self.test_set.iloc[:, :-1].values
        y = self.test_set.iloc[:, -1].values
        self.test_Y = np.eye(2)[y.astype(int)]
        val_cache = {'A0': X}

        for l in range(1, len(self.layer_sizes)):
            A_prev = val_cache[f'A{l - 1}']
            prediction = self.layers[l - 1].forward(l, A_prev)
            val_cache.update(prediction)

        A_last = val_cache[f"A{len(self.layer_sizes) - 1}"]
        val_loss, _ = self.layers[-1].categoricalCrossentropy(self.test_Y, A_last)
        return A_last, val_loss


    def train(self, log, learning_rate):
        m = self.X.shape[0]
        
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
                    self.layers[l - 1].weights -= learning_rate * dW
                    self.layers[l - 1].biases -= learning_rate * dB

            epoch_loss /= m // self.batch_size
            val_pred, val_loss = self.forward_only()
            self.save_metrics(epoch_loss, val_loss, val_pred, all_y_true, all_y_pred)
            log(epoch, epoch_loss, val_loss)


def main():
    parser = argparse.ArgumentParser(description='Predicts the cancer based on dataset')
    parser.add_argument('train_set', type=str)
    parser.add_argument('test_set', type=str)
    parser.add_argument('--layer', nargs="+", type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--loss', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)

    args = parser.parse_args()

    model = NeuronalNetwork(args.train_set, args.test_set, args.layer, args.epochs, args.loss, args.batch_size)
    model.create_layers()
    model.train(model, args.learning_rate)
    plot_loss_accuracy(model.history['loss'], model.history['val_loss'], model.history['acc'], model.history['val_acc'])


if __name__ == "__main__":
    main()
