import numpy as np
from utils import relu, softmax, relu_backward


class Layer:
    def __init__(self, previous_l, current_l, activation, weights=None, biases=None):
        if weights is not None and biases is not None:
            self.weights = weights
            self.biases = biases
        else:
            limit = np.sqrt(2.0 / previous_l)
            self.weights = np.random.uniform(-limit, limit, size=(previous_l, current_l))
            self.biases = np.full((1, current_l), 0.01)


        #print(f"Weights: {self.weights}\n Bias: {self.biases}")
        self.m_dw = np.zeros_like(self.weights)
        self.m_db = np.zeros_like(self.biases)
        self.v_dw = np.zeros_like(self.weights)
        self.v_db = np.zeros_like(self.biases)
        self.activation = activation
        self.activations = {}


    def adam_optimization(self, dw, db, t, learning_rate):
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        self.m_dw = beta1 * self.m_dw + (1 - beta1) * dw
        self.m_db = beta1 * self.m_db + (1 - beta1) * db
        self.v_dw = beta2 * self.v_dw + (1 - beta2) * (dw**2)
        self.v_db = beta2 * self.v_db + (1 - beta2) * (db**2)

        m_dw_corr = self.m_dw / (1 - beta1**t)
        m_db_corr = self.m_db / (1 - beta1**t)
        v_dw_corr = self.v_dw / (1 - beta2**t)
        v_db_corr = self.v_db / (1 - beta2**t)

        self.weights = self.weights - learning_rate * (m_dw_corr / (np.sqrt(v_dw_corr) + epsilon))
        self.biases = self.biases - learning_rate * (m_db_corr / (np.sqrt(v_db_corr) + epsilon))


    def categoricalCrossentropy(self, y_true, y_pred):
        epsilon = 1e-15
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        dA = (y_pred - y_true)
        return loss, dA


    def binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-15
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = - np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
        return loss, None


    def forward(self, l, X):
        A = X
        #print(f"SHAPE x {X.shape}, SHPAPE w {self.weights.shape}, SHAPE b {self.biases.shape}")
        Z = np.dot(X, self.weights) + self.biases
        if self.activation == 'relu':
            A = relu(Z)
        else:
            A = softmax(Z)
        self.activations[f'Z{l}'] = Z
        self.activations[f'A{l}'] = A
        return self.activations


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
        return dA_prev, dW, dB
