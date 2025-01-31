import csv
import numpy as np

train_file = 'data/train.csv'
reader = csv.reader(open(train_file, 'r'), delimiter=',')
train = np.array(list(reader))[1:, ].astype('float')
Y = train[1:, 0]
X = train[1:, ]
N = train.shape[0] - 1
D = train.shape[1]
K = int(np.max(Y)) + 1
T = np.zeros((N, K))

for i in range(N):
    for j in range(D):
        for k in range(K):
            if train[i, 0] == k:
                T[i, k] = 1

W = np.random.randn(D, K)
b = np.random.randn(K)

print(N, D, len(Y))

def softmax(a):
    c = np.max(a, axis=1, keepdims=True)
    expA = np.exp(a - c)
    return expA / expA.sum(axis=1, keepdims=True)

def predict(X, W):
    return softmax(X.dot(W) + b)

def cost(T, Y):
    return -(T * np.log(Y)).sum()

def grad(Y, T, X):
    return X.T.dot(Y - T)

def classify(Y, T, X, W, epochs, learning_rate):
    for i in range(epochs):
        Y = predict(X, W)
        W -= learning_rate * grad(Y, T, X)
    print(W)

if __name__ == '__main__':
    classify(Y, T, X, W, int(N/100), 0.01)