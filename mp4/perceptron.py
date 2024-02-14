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

def trainPerceptron(train_set, train_labels,  max_iter):
    #Write code for Mp4
    
    #each pixel is a feature so we have 3072 features for each image
    # the observation vector x^T = [x1 .... x3072] |x| = d
    # the bias vector b = [b1, b2] |b| = v = 2 (either an animal or not an animal)
    # the weight matrix W = v x d
    n_images, n_features = train_set.shape
    x = np.zeros(n_features)
    bias = np.array([0,1])
    weight = np.zeros((n_features, 2))
    
    n_rows, n_cols = train_set.shape
    
    with open('test.txt', 'w') as file:
        file.write(str(train_set) + '\n\n')
        file.write(str(train_labels) + '\n\n')
        file.write(str(max_iter) + '\n\n')
        file.write(str(train_set.shape))

    #return W, b
    return 0

def classifyPerceptron(train_set, train_labels, dev_set, max_iter):
    #Write code for Mp4

    return []



