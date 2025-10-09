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
        self.params = {key: data[key] for key in data.files if key != 'topology'}
        layers = data['topology']
        input_size = layers[0]
        hidden_layers = list(layers[1:-1])
        self.test_set = test_set
        if 'id' in test_set.columns:
            self.test_set = self.test_set.drop(columns=['id'])
        Y = self.test_set.iloc[:, 0].values
        self.test_Y = np.eye(2)[Y.astype(int)]
        self.configure(input_size, self.test_Y, hidden_layers, output_size=2)


def main():
    parser = argparse.ArgumentParser(description="Predicts")
    parser.add_argument("--weights", type=str)
    parser.add_argument("--preprocessor", type=str)
    parser.add_argument("--input_set", type=str)
    args = parser.parse_args()

    with open(args.preprocessor, "rb") as f:
        pre = pickle.load(f)
    try:
        data = open_file(args.input_set)
        data = pre.transform(data)
    except Exception as e:
        print(e)
        exit(1)

    predict = Predict(args.weights, data)
    predict.create_layers(params=predict.params)
    pred, loss  = predict.forward_only(predict.test_set, predict.layers[-1].binary_cross_entropy)
    accuracy = np.mean(np.argmax(pred, axis=1) == np.argmax(predict.test_Y, axis=1))
    print(f"Loss: {loss:.2f}, Accuracy: {accuracy * 100:.2f} %")


if __name__ == "__main__":
    main()
