import argparse
import numpy as np
from base import BaseNetwork
from layer import Layer
from utils import open_file


class Predict(BaseNetwork):
    def __init__(self, data_set, test_set):
        try:
            data = np.load(data_set)
        except Exception as e:
            print(e)
            exit(1)
        self.num_values = data.iloc[:,2:]
        self.test_set = open_file(test_set)
        self.bias = data["b"]
        self.weight = data["w"]


    def binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-12
        m = len(self.num_values)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = - np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
        print("Loss: ", loss)





    def main():
        parser = argparse.ArgumentParser(description="Predicts")
        parser.add_argument("input_set", type=str)
        args = parser.parse_args()

        predict = Predict(args.input_set)
        predict.forward_only()

    if __name__ == "__main__":
        main()
