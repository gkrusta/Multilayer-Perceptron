import numpy as np


class Layer:
    def __init__(self, previous_l, current_l, activation):
        limit = np.sqrt(6 / current_l)
        self.weights = np.random.uniform(-limit, limit, size=(previous_l, current_l))
        self.biases = np.zeros((1, current_l))
        print(f"Weights: {self.weights}\n Bias: {self.biases}")
        self.activation = activation

    
    def forward(self, input):
        ...
    
    def backward(self, grad_output, learning_rate):
        ...