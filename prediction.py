import argparse
import pickle
import numpy as np
from base import BaseNetwork
from utils import open_file


class Predict(BaseNetwork):
    def __init__(self, data_set, test_set):
        super().__init__()
        try:
            data = np.load(data_set, allow_pickle=True)
        except Exception as e:
            print(e)
            exit(1)
        print("INPUT SET ")
        layer = data["topology"]
        input_size = test_set.drop(columns=['diagnosis']).shape[1]
        self.y_true = test_set.iloc[:, -1].values
        self.configure(input_size, self.y_true, layer, output_size=2)


def main():
    parser = argparse.ArgumentParser(description="Predicts")
    parser.add_argument("weights", type=str)
    parser.add_argument("preprocessor", type=str)
    parser.add_argument("input_set", type=str)
    args = parser.parse_args()

    with open(args.preprocessor, "rb") as f:
        pre = pickle.load(f)
    try:
        data = open_file(args.input_set, header_in_file=False)
        data = pre.transform(data)
    except Exception as e:
        print(e)
        exit(1)

    predict = Predict(args.weights, data)
    predict.create_layers()
    pred, loss  = predict.forward_only(data, predict.layers[-1].binary_cross_entropy)
    accuracy = np.mean(np.argmax(pred, axis=1) == predict.y_true)
    print(f"Loss: {loss}, Accuracy: {accuracy} %")


if __name__ == "__main__":
    main()
