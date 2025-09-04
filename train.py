from numpy import np
import argparse


epoch = 10
learning_rate = 0.1



def train():
    weighted_sum = np.dot(x, weights)
    prediction = sigmoid(weighted_sum)

    # loss = MSE
    # backpropogation
    # gradients for weight and bias
    # actulize weights and bias using leanring rate


def main():
    # parser = argparse.ArgumentParser(description='Predicts the cancer based on dataset')
    # parser.add_argument('dataset')
    # parser.add_argument('--layer', action='store_true')
    # parser.add_argument('--epochs', action='store_true')
    # parser.add_argument('--los', action='store_true')
    # parser.add_argument('--batch_size', action='store_true')
    # parser.add_argument('--learning_rate', action='store_true')

    # args = parser.parse_args()

    # df = parse(args.dataset)
    # if args.plots:
    #   func(df)
    

if __name__ == "__main__":
    main()
