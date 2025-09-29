import numpy as np
from layer import Layer


class BaseNetwork:
    def __init__(self):
        self.layers = []


    def configure(self, input_size, test_Y, hidden_layers, output_size=2):
        self.input_size = input_size
        self.test_Y = test_Y
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]


    def create_layers(self, activation='relu', output_activation='softmax'):
        if len(self.hidden_layers) == 1:
            self.layer_sizes.insert(2, self.hidden_layers[0])

        for i in range(1, len(self.layer_sizes)):
            act = output_activation if i == len(self.layer_sizes) - 1 else activation
            layer = Layer(self.layer_sizes[i - 1], self.layer_sizes[i], act)
            self.layers.append(layer)


    def forward_only(self, test_set, loss_fn):
        X = test_set.iloc[:, :-1].values
        val_cache = {'A0': X}

        for l in range(1, len(self.layer_sizes)):
            A_prev = val_cache[f'A{l - 1}']
            prediction = self.layers[l - 1].forward(l, A_prev)
            val_cache.update(prediction)

        A_last = val_cache[f"A{len(self.layer_sizes) - 1}"]
        loss, _ = loss_fn(self.test_Y, A_last)
        return A_last, loss
