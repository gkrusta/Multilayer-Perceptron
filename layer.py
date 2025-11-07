import numpy as np
from utils import relu, softmax, relu_backward


class Layer:
    """Represents fully dense layer in neural network."""
    def __init__(self, previous_l, current_l, activation, weights=None, biases=None):
        # initialize  weights & biases
        if weights is not None and biases is not None:
            self.weights = weights
            self.biases = biases
        else:
            # He initialization for Relu layers
            self.weights = np.random.randn(previous_l, current_l) * np.sqrt(2.0 / previous_l)
            self.biases = np.zeros((1, current_l))

        # Adam moment vectors (m = mean, v = variance)
        self.m_dw = np.zeros_like(self.weights)
        self.m_db = np.zeros_like(self.biases)
        self.v_dw = np.zeros_like(self.weights)
        self.v_db = np.zeros_like(self.biases)

        self.activation = activation
        self.activations = {} # stores intermediate Z and A for backprop


    def adam_optimization(self, dw, db, t, learning_rate):
        """Applies the Adam optimization algorithm to update weights and biases."""
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

        self.weights -= learning_rate * (m_dw_corr / (np.sqrt(v_dw_corr) + epsilon))
        self.biases = self.biases - learning_rate * (m_db_corr / (np.sqrt(v_db_corr) + epsilon))


    def categorical_cross_entropy(self, y_true, y_pred):
        """Computes categorical cross entropy loss and its gradient. The two output probabilities must sum to 1."""
        epsilon = 1e-15
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        dA = (y_pred - y_true) # derivative of cross entropy with softmax
        return loss, dA


    def binary_cross_entropy(self, y_true, y_pred):
        """"Computes binary cross entropy loss and its gradient. Takes 1 probability only."""
        epsilon = 1e-15
        m = y_true.shape[0]
        y_true = y_true[:, 1]
        y_pred = y_pred[:, 1]
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = - np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
        dA = (y_pred - y_true)
        return loss, dA


    def forward(self, l, X):
        """Performs forward propagation for this layer.
        Z = XW + b
        A = activation(Z)"""
        A = X
        Z = np.dot(X, self.weights) + self.biases
        if self.activation == 'relu':
            A = relu(Z)
        else:
            A = softmax(Z)
        self.activations[f'Z{l}'] = Z
        self.activations[f'A{l}'] = A
        return self.activations


    def backward(self, dA, layer, l):
        """Performs backward propagation for this layer."""
        m = dA.shape[0]
        dA_prev = layer[f'A{l - 1}']

        if self.activation == 'relu':
            dZ = dA * relu_backward(layer[f'Z{l}'])
        elif self.activation == 'softmax':
            dZ = dA
        else:
            raise Exception('Non-supported activation function')
        dW = np.dot(dA_prev.T, dZ) / m # Gradient of weights
        dB = np.sum(dZ, axis=0, keepdims=True) / m # Gradients of biases
        dA_prev = np.dot(dZ, self.weights.T) # Gradient w.r.t. previous layer's activation
        return dA_prev, dW, dB
