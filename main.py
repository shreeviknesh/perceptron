from perceptron import train, predict

# We can use any set of Xs and Ys to fit the perceptron
# A perceptron training will only converge if the relationship between X&Y is linear
# (i.e., linearly separable data)

# Data given for the assignment
X = [[0.25, 0.353],
     [0.25, 0.471],
     [0.5, 0.353],
     [0.5, 0.647],
     [0.75, 0.705],
     [0.75, 0.882],
     [1, 0.705],
     [1, 1]]
Y = [0, 1, 0, 1, 0, 1, 0, 1]

W, b = train(X, Y, learning_rate=0.05)

# Checking if the converged values are correct
print('Prediction is correct?', predict(X, W, b) == Y)
