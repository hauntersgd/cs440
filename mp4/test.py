import reader
import perceptron as perceptron
import numpy as np
import mp4 as mp4

# with open('test.txt', 'w') as file:
    #             file.write(str(prod0) + "\n\n")
    #             file.write(str(prod1) + "\n\n")
    #             file.write(str(classes) + "\n")
    #             file.write(str(len(dev_set)) + "\n")
    #             file.write(str(len(classes)))

# with open('traintest.txt', 'w') as file:
    #     file.write(str(weight_diff) + '\n\n')
    #     file.write(str(bias) + '\n\n')

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

perceptron.classifyPerceptron(train_set, train_labels, train_set, max_iter)


