import numpy as np
from utils import relu, softmax, relu_backward


class Layer:
    def __init__(self, previous_l, current_l, activation):
        limit = np.sqrt(6 / current_l)
        self.weights = np.random.uniform(-limit, limit, size=(previous_l, current_l))
        self.biases = np.zeros((1, current_l))
        print(f"Weights: {self.weights}\n Bias: {self.biases}")
        self.activation = activation
        self.activations = {}



    def forward(self, l, X):
        A = X
        Z = np.dot(X, self.weights) + self.biases
        if self.activation == 'relu':
            A = relu(Z)
        else:
            A = softmax(Z)
        self.activations[f'Z{l}'] = Z
        self.activations[f'A{l}'] = A

        return self.activations


    def categoricalCrossentropy(self, y_true, y_pred):
        epsilon = 1e-12
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        dA = (y_pred - y_true)

        return loss, dA


    def backward(self, dA, layer, l):
        m = dA.shape[0]
        dA_prev = layer[f'A{l - 1}']

        if self.activation == 'relu':
            dZ = dA * relu_backward(layer[f'Z{l}'])
        elif self.activation == 'softmax':
            dZ = dA
        else:
            raise Exception('Non-supported activation function')
        dW = np.dot(dA_prev.T, dZ) / m
        dB = np.sum(dZ, axis=0, keepdims=True) / m
        dA_prev = np.dot(dZ, self.weights.T)
        print("1 DW: ", dW)
        return dA_prev, dW, dB
