import numpy as np
import argparse
from utils import open_file
from layer import Layer

epoch = 10
learning_rate = 0.01


class NeuronalNetwork:
    def __init__(self, train_set, layer):
        df = open_file(train_set)
        self.numer_of_inputs = df.drop(columns=['diagnosis']).shape[1]
        self.hiden_layers = layer
        self.train_set = train_set
        self.output_size = 2


# def categorical_cross_entropy()


    def create_layers(self, activation='relu', output_activation='softmax'):
        self.layer_sizes = [self.numer_of_inputs] + self.hiden_layers + [self.output_size]
        if len(self.hiden_layers) == 1:
            self.layer_sizes.insert(2, self.hiden_layers[0])

        self.layers = []
        for i in range(1, len(self.layer_sizes)):
            act = output_activation if i == len(self.layer_sizes) - 1 else activation
            layer = Layer(self.layer_sizes[i - 1], self.layer_sizes[i], act)
            self.layers.append(layer)



    def train(x, weights):
        weighted_sum = np.dot(x, weights)
        prediction = sigmoid(weighted_sum)

        # loss = MSE
        # backpropogation
        # gradients for weight and bias
        # actulize weights and bias using leanring rate



def main():
    parser = argparse.ArgumentParser(description='Predicts the cancer based on dataset')
    parser.add_argument('train_set', type=str)
    parser.add_argument('--layer', nargs="+", type=int)
    #parser.add_argument('test_set', type=str, required=True)
    # parser.add_argument('--epochs', action='store_true')
    # parser.add_argument('--los', action='store_true')
    # parser.add_argument('--batch_size', action='store_true')
    # parser.add_argument('--learning_rate', action='store_true')

    args = parser.parse_args()

    model = NeuronalNetwork(args.train_set, args.layer)
    model.create_layers()
    

if __name__ == "__main__":
    main()
