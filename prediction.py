import argparse
import pickle
import numpy as np
from base import BaseNetwork
from utils import open_file


class Predict(BaseNetwork):
    def __init__(self, data_set, test_set):
        try:
            data = np.load(data_set)
        except Exception as e:
            print(e)
            exit(1)
        test_set = open_file(test_set)
        self.num_values = data.iloc[:,2:]
        bias = data["b"]
        weight = data["w"]
        layer = data["topology"]
        input_size = test_set.drop(columns=['diagnosis']).shape[1]
        y_true = test_set.iloc[:,-1].values
        self.configure(input_size, y_true, layer, output_size=2)


    def binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-12
        m = len(self.num_values)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = - np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
        print("Loss: ", loss)


    def main():
        parser = argparse.ArgumentParser(description="Predicts")
        parser.add_argument("weights", type=str)
        parser.add_argument("preprocessor", type=str)
        parser.add_argument("input_set", type=str)
        args = parser.parse_args()

        with open(args.preprocessor, "rb") as f:
            pre = pickle.load(f)
        try:
            data = np.load(args.input_set)
            data = pre.transform(data)
        except Exception as e:
            print(e)
            exit(1)

        predict = Predict(args.weights, args.input_set)
        predict.create_layers()
        predict.forward_only()


    if __name__ == "__main__":
        main()
