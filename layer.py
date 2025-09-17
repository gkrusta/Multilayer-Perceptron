import numpy as np
from utils import relu, softmax

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

        
    
    def backward(self, grad_output, learning_rate):
        ...