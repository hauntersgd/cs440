# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020



import numpy as np
eta = 0.01 # learning rate
epsilon = 1e-10

def trainPerceptron(train_set, train_labels,  max_iter):
    #Write code for Mp4
    
    #each pixel is a feature so we have 3072 features for each image
    # the observation vector x^T = [x1 .... x3072] |x| = d
    # the bias vector b = [b1, b2] |b| = v = 2 (either an animal or not an animal)
    # the weight matrix W = v x d
    # initialize weights and biases to zero
    n_images, n_features = train_set.shape
    x = np.zeros(n_features)
    bias = np.zeros(2)
    weight_diff = np.zeros((2, n_features))
    
    for images in range(max_iter):
        # n_images vectors, with n_features
        for image_vector_x, y in zip(train_set, train_labels):
            prod0 = np.dot(weight_diff[0], image_vector_x) + bias[0]
            prod1 = np.dot(weight_diff[1], image_vector_x) + bias[1]
            y_hat = 0
            if prod0 > prod1:
                y_hat = 0
            elif prod1 >= prod0:
                y_hat = 1
    
            if y_hat == y and y == 0:
                weight_diff[0] = weight_diff[0] - (eta * image_vector_x)
                bias[0] -= eta
            elif y_hat == y and y == 1:
                weight_diff[1] = weight_diff[1] - (eta * image_vector_x)
                bias[1] -= eta
            else:
                continue

    return weight_diff, bias

def classifyPerceptron(train_set, train_labels, dev_set, max_iter):
    #Write code for Mp4
    weights, bias = trainPerceptron(train_set, train_labels, max_iter)
    classes = []
    for image in dev_set:
        prod0 = np.dot(weights[0], image) + bias[0]
        prod1 = np.dot(weights[1], image) + bias[1]
        y_hat = 0
        if prod0 - prod1 > 0:
            y_hat = 1
        elif prod0 - prod1 <= 0:
            y_hat = 0
        classes.append(y_hat)

    # return classes
    return classes
