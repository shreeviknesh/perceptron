import numpy as np

# Defining the activation function
def activation_function(value):
    if value >= 0:
        return 0
    else:
        return 1

# The main train loop
def train(X, Y, learning_rate=0.01, max_iterations=500):
    X = np.array(X)
    Y = np.array(Y)

    num_inputs = len(X)

    # Initializing the weights and the bias
    W = np.random.rand(len(X[0]))
    b = np.random.random()

    # Learning Loop
    iteration = 1
    while True:
        if iteration > max_iterations:
            print(f'Iterations limit reached - {iteration-1}')
            print('Weight values:', W)
            break

        flag = True
        for i in range(num_inputs):
            net = np.dot(W, X[i]) + b
            Yhat = activation_function(net)

            if Y[i] > Yhat:
                W = W - learning_rate * X[i]
                flag = False
                break
            elif Y[i] < Yhat:
                W = W + learning_rate * X[i]
                flag = False
                break

        # Checking if the values have converged
        if flag == True:
            print(f'Converged after {iteration} iterations')
            print('Weight values:', *W, b)
            break

        iteration += 1

    return (W, b)

# A function that predicts the y values given X values, weights and bias
def predict(X, W, b):
    Y = []
    for i in range(len(X)):
        net = np.dot(W, X[i]) + b
        Yhat = activation_function(net)
        Y.append(Yhat)
    return Y