'''
Written by Ben McCoy, Jan 2020

Given that my first aattempt at a NN from scratch does not seem to learn, I
found a model for an even simple NN that I want to try.

Based on article:
becominghuman.ai/lets-build-a-simple-neural-net-f4474256647f
'''

import pandas as pd
import numpy as np

class SimpleNeuralNetwork(object):
    # init the weight matrix
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights=[]

    def __sigmoid(self, x, deriv=False):
        # uses a sigmoid activation function
        # takes two parameters, x (a list) and deriv (default is false)
        # if deriv is true, the derivative of the sigmoid function is returned
        if deriv == True:
            return x * (1-x)
        return 1 / (1 + np.exp(-x))

    def predict(self, x):
        # take an input array (X) and return the product of X and the weight
        # matrix
        predicted = np.dot(x, self.synaptic_weights)
        return self.__sigmoid(predicted)

    def train(self, file, X, y, iterations):
        dim = file.shape
        self.synaptic_weights = 2 * np.random.random((dim[1] - 1, 1)) - 1

        for i in range(iterations):
            output = self.predict(X)
            error = y - output

            adjustment = np.dot(X.T, error * self.__sigmoid(output, deriv=True))
            self.synaptic_weights += adjustment


if __name__ == "__main__":
    data = pd.read_csv('file.csv')
    # print(data)

    X = data.iloc[:, 0:4].values
    y = data.iloc[:, [4]].values
    # print(X)
    # print(y)

    number_of_iterations = 6000
    clf = SimpleNeuralNetwork()

    clf.train(data, X, y, number_of_iterations)

    prediction = np.array([0,1,1,0])
    res = clf.predict(prediction)[0]

    if res >= 0.5:
        print('Prediction:', 1)
    else:
        print('Prediction:', 0)
