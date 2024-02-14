import reader
import perceptron as perceptron
import numpy as np
import mp4 as mp4

X, y, decision_curve = mp4.classification_problem(batch_size=100)
affine = True  # test bias term
if affine:
    translation = np.array([3, -3])
    X += translation
train_size = int(len(X) / 2)
train_set = X[:train_size]
test_set = X[train_size:]
train_labels = y[:train_size]
test_labels = y[train_size:]
lrate = 1
max_iter = 10

perceptron.trainPerceptron(train_set, train_labels, max_iter)


