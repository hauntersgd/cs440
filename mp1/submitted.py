'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
from collections import Counter

def marginal_distribution_of_word_counts(texts, word0):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the word that you want to count

    Output:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
    '''
    #raise RuntimeError("You need to write this part!")
    Pmarginal = np.zeros(0)
    count = 0 
    for text in texts:
        for word in text:
            if word == word0:
              count += 1
        if count >= Pmarginal.size:
            Pmarginal.resize(count+1)
        Pmarginal[count] += 1
        count = 0
    
    Pmarginal = Pmarginal / len(texts)
  
    return Pmarginal
    
def conditional_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word that you want to count
    word1 (str) - the second word that you want to count

    Outputs: 
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0), where
      X0 is the number of times that word0 occurs in a document
      cX0-1 is the largest value of X0 observed in the provided texts
      X1 is the number of times that word1 occurs in a document
      cX1-1 is the largest value of X0 observed in the provided texts
      CAUTION: If P(X0=x0) is zero, then P(X1=x1|X0=x0) should be np.nan.
    '''
    #raise RuntimeError("You need to write this part!")
    matrix = np.zeros((0, 0))
    count0 = 0
    count1 = 0
    for text in texts:
        for word in text:
            if word == word0:
                count0 += 1
            if word == word1:
                count1 += 1
        rows, cols = matrix.shape
        rowsize = max(count0 + 1, rows)
        colsize = max(count1 + 1, cols)

        matrixnew = np.zeros((rowsize, colsize))
        matrixnew[:rows, :cols] = matrix
        matrixnew[count0, count1] += 1
        matrix = matrixnew
        count0 = 0
        count1 = 0
    matrix = matrix / np.sum(matrix) # convert to a joint distribution

    p0 = marginal_distribution_of_word_counts(texts, word0)

    for i in range(matrix.shape[0]):
        if p0[i] == 0:
          matrix[i] = np.nan
        else:
          matrix[i] /= p0[i]

  
    Pcond = np.zeros((0,0))
    Pcond = matrix

    return Pcond

def joint_distribution_of_word_counts(Pmarginal, Pcond):
    '''
    Parameters:
    Pmarginal (numpy array of length cX0) - Pmarginal[x0] = P(X0=x0), where
    Pcond (numpy array, shape=(cX0,cX1)) - Pcond[x0,x1] = P(X1=x1|X0=x0)

    Output:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
      CAUTION: if P(X0=x0) then P(X0=x0,X1=x1)=0, even if P(X1=x1|X0=x0)=np.nan.
    '''
    #raise RuntimeError("You need to write this part!")
    Pjoint = Pcond
    for i in range(Pcond.shape[0]):
        if Pmarginal[i] == 0:
            Pjoint[i] = 0
        else:
          Pjoint[i] = Pcond[i] * Pmarginal[i]
    return Pjoint

def mean_vector(Pjoint):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    
    Outputs:
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    '''
    #raise RuntimeError("You need to write this part!")
    X0 = np.zeros(Pjoint.shape[0])
    X1 = np.zeros(Pjoint.shape[1])

    for i in range(Pjoint.shape[0]):
        rowsum = np.sum(Pjoint[i])
        X0[i] = rowsum

    for j in range(Pjoint.shape[1]):
        colsum = np.sum(Pjoint[:, j])
        X1[j] = colsum
    
    mu = np.zeros(2)
    for idx, val in enumerate(X0):
        mu[0] += idx * val
    for idx, val in enumerate(X1):
        mu[1] += idx * val
  
    return mu

def variance_vector(Pjoint, mu):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    
    Outputs:
    var (numpy array, length 2) - the variance of the vector [X0, X1]
    '''
    #raise RuntimeError("You need to write this part!")
    X0 = np.zeros(Pjoint.shape[0])
    X1 = np.zeros(Pjoint.shape[1])

    for i in range(Pjoint.shape[0]):
        rowsum = np.sum(Pjoint[i])
        X0[i] = rowsum

    for j in range(Pjoint.shape[1]):
        colsum = np.sum(Pjoint[:, j])
        X1[j] = colsum
    
    var = np.zeros(2)
    for idx, val in enumerate(X0):
        var[0] += idx * idx * val # E[x0^2]
    for idx, val in enumerate(X1):
        var[1] += idx * idx * val # E[x1^2]
    
    var[0] -= (mu[0] * mu[0])
    var[1] -= (mu[1] * mu[1])
  
    return var

def covariance_matrix(Pjoint, mu):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    mu (numpy array, length 2) - the mean of the vector [X0, X1]
    
    Outputs:
    Sigma (numpy array, shape=(2,2)) - matrix of variance and covariances of [X0,X1]
    '''

    #raise RuntimeError("You need to write this part!")

    # Var(x) = E[X^2] - E[x]^2

    # [Var(x0)       Cov(x0,x1)  ]
    # [Cov(x1, x0)   Var(x1)     ]

    Exy = 0
    cov = 0

    Sigma = np.zeros((2,2))
    var = variance_vector(Pjoint, mu)
    Sigma[0][0] = var[0]
    Sigma[1][1] = var[1]

    for x0 in range(Pjoint.shape[0]):
        for x1 in range(Pjoint.shape[1]):
            Exy += x0 * x1 * Pjoint[x0][x1]

    cov = Exy - (mu[0] * mu[1])

    Sigma[0][1] = cov
    Sigma[1][0] = cov

    return Sigma

def distribution_of_a_function(Pjoint, f):
    '''
    Parameters:
    Pjoint (numpy array, shape=(cX0,cX1)) - Pjoint[x0,x1] = P(X0=x0, X1=x1)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       may be any hashable value (number, string, or even a tuple).

    Output:
    Pfunc (Counter) - Pfunc[z] = P(Z=z)
       Pfunc should be a collections.defaultdict or collections.Counter, 
       so that previously unobserved values of z have a default setting
       of Pfunc[z]=0.
    '''
    #raise RuntimeError("You need to write this part!")

    Pfunc = Counter()
    for x0 in range(Pjoint.shape[0]):
        for x1 in range(Pjoint.shape[1]):
            z = f(x0, x1)
            Pfunc[z] += Pjoint[x0][x1]

    return Pfunc
    
