A simple Python machine learning application for reading images of digits.

## Requirements

- Python 3.x
- NumPy
- CSV

## Setup

1. Clone the repository.
2. Install the required packages:
    ```sh
    pip install numpy
    ```

## Usage

1. Ensure the training data is available in the [train.csv](http://_vscodecontentref_/4) file.
2. Run the [digit-recognizer.py](http://_vscodecontentref_/5) script:
    ```sh
    python digit-recognizer.py
    ```

## Functions

- [softmax(a)](http://_vscodecontentref_/6): Computes the softmax of the input array `a`.
- [predict(X, W)](http://_vscodecontentref_/7): Predicts the output using the input [X](http://_vscodecontentref_/8) and weights [W](http://_vscodecontentref_/9).
- [cost(T, Y)](http://_vscodecontentref_/10): Computes the cost function.
- [grad(Y, T, X)](http://_vscodecontentref_/11): Computes the gradient of the cost function.
- [classify(Y, T, X, W, epochs, learning_rate)](http://_vscodecontentref_/12): Trains the logistic regression model.

## License

This project is licensed under the MIT License.
