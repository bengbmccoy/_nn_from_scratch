'''
Written by Ben McCoy July 2019
Thanks to Piotr Skalski for his article in Towards Data Science

This script is where the neural network will be written.

To begin with, we will initialise the neural netowrk layers and archtecture . - DONE
Then we will initiate the parameter values for each layer, these weights cannot
begin as the same values or we end up with the breaking symmetry issue. - DONE
Then we will outline the activation functions that we can use as well as their
derivatives to be used for back propogation. - DONE
Then we will colde the single layer forward propogation step. - DONE
Then the full layer forward propogation step, which will use a dictionary to
act as memory so that the loss function and back propogation can be done. - DONE
Then a cost function will be written, as well as a accuracy function.
Then a single layer back propogation step will be written.
Then a full layer back propogation step.
Then the parameters will be updated.
Then a training function.
'''

import numpy as np

nn_architecture = [
{'input_dim': 2, 'output_dim': 4, 'activation': 'relu'},
{'input_dim': 4, 'output_dim': 6, 'activation': 'relu'},
{'input_dim': 6, 'output_dim': 6, 'activation': 'relu'},
{'input_dim': 6, 'output_dim': 4, 'activation': 'relu'},
{'input_dim': 4, 'output_dim': 1, 'activation': 'sigmoid'},
]

def main():

    param_values = init_layers(nn_architecture)
    # print(param_values)
    print('param values initiated')

def init_layers(nn_architecture, seed=99):
    '''This function will take the neural netowrk archtecture and a seed
    variable and will initialise the matrix weights and bias vector for each
    layer in the neural netwrok. These values will be stored in the param_values
    dictionary to be passed on and optimised.'''

    np.random.seed(seed)
    # initiates a random seed that is consistent in this function block
    num_layers = len(nn_architecture)
    # returns the number of layers in the nn_architecture
    param_values = {}
    # stores the parameter values that we initiate

    for idx, layer in enumerate(nn_architecture):
    # iterates through the layers of the nn
        layer_idx = idx + 1
        # layers index starts at 1 and not 0
        layer_imput_size = layer['input_dim']
        # returns number of inputs in layer
        layer_output_size = layer['output_dim']
        # returns number of outputs in layer

        param_values['W' + str(layer_idx)] = np.random.randn(layer_output_size, layer_imput_size) * 0.1
        # initiates the weights of each connection in the layer
        param_values['b' + str(layer_idx)] = np.random.randn(layer_output_size, 1) * 0.1
        # initiates the bias of each node in the layer

    return param_values

def sigmoid(Z):
    '''The Sigmoid activation function'''
    return 1/(1+np.exp(-Z))

def relu(Z):
    '''The reLU activation function'''
    return np.maximum(0,Z)

def sigmoid_backwards(dA, Z):
    '''The derivative of the Sigmoide activation function'''
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backwards(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;

def single_layer_forward_prop(A_prev, W_curr, b_curr, activation='relu'):
    '''This function takes the input signal from the previous layer and computes
    the transformation Z using the weights W, the bias B and the previous values
    of A. It then applies the selected activation function to matrix Z'''

    Z_curr = np.dot(W_curr, A_prev) + b_curr
    # calculates Z, the matrix of intermediate values at the nodes of a layer
    # These will be the input value for the activation function

    if activation is 'relu':
        activation_func = relu
    elif activation is 'sigmoid':
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')

    return activation_func(Z_curr), Z_curr
    # returns valculated activation A and the intermediate Z matrix

def full_forward_propogation(X, param_values, nn_architecture):
    '''This function performs a full forward propogation step using the single
    layer propogation function. The function takes the input matrix X, as well
    as the current param values. Returns a prediction vector and a dictionary
    contianing intermediate values'''

    memory = {}
    # stores the values required for a backward propogation step
    A_curr = X
    # the X vector is the activation for layer 0

    for idx, layer in enumerate(nn_architecture):
    # iterates through the layers of the nn
        layer_idx = idx + 1
        # layers index starts at 1 and not 0
        A_prev = A_curr
        # transfers the activation from the previous iteration

        activ_function_curr = layer['activation']
        # find the activation function of the current layer
        W_curr = param_values['W' + str(layer_idx)]
        # find the current weights of the current layer
        b_curr = param_values['b' + str(layer_idx)]
        # find the current biases of the current layer
        A_curr, Z_curr = single_layer_forward_prop(A_prev, W_curr, b_curr, activ_function_curr)
        # calculate the activation for the current layer

        memory['A' + str(idx)] = A_prev
        # save the calculated values of A in memory as a python dict
        memory['Z' + str(layer_idx)] = Z_curr
        # Save the calculated values of Z in memory as a python dict

    return A_curr, memory

def get_cost_value(Y_hat, Y):
    '''The cost function in this project is a binary crossentropy function
    which is used to test the classification of points between two classes.'''

    m = Y_hat.shape[1]
    # Get the number of training examples
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    # calculate the binary crossentropy cost
    return np.squeeze(cost)
    # Removes the single dimensional entries from the shape of an array

def get_accuracy_value(Y_hat, Y):
    '''A function that determines the accuracy of a prediction vector'''

    Y_hat_ = convert_prob_to_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()

def convert_prob_to_class(probs):
    '''A function that converts the probability of a prediction into a defined
    classification of 1 or 0'''

    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def single_layer_back_prop(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation='relu'):
    '''Back propogation of a single layer'''

    m = A_prev.shape[1]
    # Get the number of training examples

    if activation is 'relu':
        backward_activation_func = relu_backwards
    elif activation is 'sigmoid':
        backward_activation_func = sigmoid_backwards
    else:
        raise Exception('Non-supported activation function')
    # determines the correct activation function to use

    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    # calculation of the change in the matrix of node values
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    # calculate the chnage in the weights matrix
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    # calculate the change in the bias vector
    dA_prev = np.dot(W_curr.T, dZ_curr)
    # calculate the change of the activation matrix

    return dA_prev, dW_curr, db_curr
    # return the change in actiavtion values, weights and bias vector

def full_backward_propogation(Y_hat, Y, memory, param_values, nn_architecture):
    '''A full backward propogation, '''

    grads_values = {}
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)

    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
    # initiation of gradient descent algorithm

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        # iterate through the layers of the NN backwards

        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer['activation']
        # find the layer's activations function

        dA_curr = dA_prev

        A_prev = memory['A' + str(layer_idx_prev)]
        Z_curr = memory['Z' + str(layer_idx_curr)]
        W_curr = param_values['W' + str(layer_idx_curr)]
        b_curr = param_values['b' + str(layer_idx_curr)]

        dA_prev, dW_curr, db_curr = single_layer_back_prop(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

        grads_values['dW' + str(layer_idx_curr)] = dW_curr
        grads_values['db' + str(layer_idx_curr)] = db_curr

    return grads_values

def update(param_values, grads_values, nn_architecture, learning_rate):
    '''This function updates the patameter values using gradient optimisation
    The optimisation algorithm being used here is:
    Wl = Wl - learning_rate * dWl
    bl = bl - learning_rate * dbl'''

    for layer_idx, layer in enumerate(nn_architecture, 1):
        # iterate over the NN layers

        param_values['W' + str(layer_idx)] -= learning_rate * grads_values['dW' + str(layer_idx)]
        param_values['b' + str(layer_idx)] -= learning_rate * grads_values['db' + str(layer_idx)]

    return param_values;

def train(X, Y, nn_architecture, epochs, learning_rate):
    ''''''

    param_values = init_layers(nn_architecture, 2)
    # initiate the NN parameters
    cost_history = []
    accuracy_history = []

    for i in range(epochs):
        # perform calcualtions for number of epochs

        Y_hat, cashe = full_forward_propogation(X, param_values, nn_architecture)
        # step forward

        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)
        # calculate the cost and accuracy metics and save them in a list

        grads_values = full_backward_propogation(Y_hat, Y, cashe, param_values, nn_architecture)
        # step backward

        param_values = update(param_values, grads_values, nn_architecture, learning_rate)
        # update model's state

        if (i % 50 == 0):
            if(verbose):
                print('Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}'.format(i, cost, accuracy))

            if (callback is not None):
                callback(i, param_values)

    return param_values, cost_history, accuracy_history

main()
