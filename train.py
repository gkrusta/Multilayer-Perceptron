from numpy import np


epoch = 10
learning_rate = 0.1



def train():
    weighted_sum = np.dot(x, weights)
    prediction = sigmoid(weighted_sum)

    # loss = MSE
    # backpropogation
    # gradients for weight and bias
    # actulize weights and bias using leanring rate
