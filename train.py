from functools import cache
import numpy as np
import argparse
from utils import open_file
from layer import Layer


class NeuronalNetwork:
    def __init__(self, train_set, test_set, layer, epochs):
        self.df = open_file(train_set)
        self.test_set = open_file(test_set)
        self.numer_of_inputs = self.df.drop(columns=['diagnosis']).shape[1]
        self.hiden_layers = layer
        self.X = self.df.iloc[:, :-1].values
        y = self.df.iloc[:, -1].values
        self.Y = np.eye(2)[y.astype(int)]
        self.output_size = 2
        self.epochs = epochs
        self.cache = {}
        self.first = True


    def __call__(self, epoch, loss, val_loss):
        if self.first:
            print("x_train shape : ", self.df.shape)
            print("x_valid shape : ", self.test_set.shape)
            self.first = False
        print(f"epoch {epoch:02d}/{self.epochs} - loss: {loss:.8f} - val_loss: {val_loss:.8f}")


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
        Y = np.eye(2)[y.astype(int)]
        val_cache = {'A0': X}

        for l in range(1, len(self.layer_sizes)):
            A_prev = val_cache[f'A{l - 1}']
            prediction = self.layers[l - 1].forward(l, A_prev)
            val_cache.update(prediction)

        A_last = val_cache[f"A{len(self.layer_sizes) - 1}"]
        val_loss, _ = self.layers[-1].categoricalCrossentropy(Y, A_last)
        return val_loss


    def train(self, log, learning_rate):
        self.cache = {'A0': self.X}
        for epoch in range(1, self.epochs + 1):
            for l in range(1, len(self.layer_sizes)):
                A_prev = self.cache[f'A{l - 1}']
                prediction = self.layers[l - 1].forward(l, A_prev)
                self.cache.update(prediction)
            loss, dA = self.layers[l - 1].categoricalCrossentropy(self.Y, self.cache[f'A{l}'])

            for l in reversed(range(l, len(self.layer_sizes))):
                dA_prev, dW, dB = self.layers[l - 1].backward(dA, self.layers[l - 1], l)
                dA = dA_prev
                print("2 DB: ", dB.shape)
                self.layers[l - 1].weights -= learning_rate * dW
                self.layers[l - 1].biases -= learning_rate * dB
            val_loss = self.forward_only()
            log(epoch, loss, val_loss)


def main():
    parser = argparse.ArgumentParser(description='Predicts the cancer based on dataset')
    parser.add_argument('train_set', type=str)
    parser.add_argument('--layer', nargs="+", type=int)
    parser.add_argument('test_set', type=str)
    parser.add_argument('--epochs', type=int)
    # parser.add_argument('--loss', action='store_true')
    # parser.add_argument('--batch_size', action='store_true')
    parser.add_argument('--learning_rate', type=float)

    args = parser.parse_args()

    model = NeuronalNetwork(args.train_set, args.test_set, args.layer, args.epochs)
    model.create_layers()
    model.train(model, args.learning_rate)
    


if __name__ == "__main__":
    main()
