'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
'''
Note:
For grading purpose, all bigrams are represented as word1*-*-*-*word2

Although you may use tuple representations of bigrams within your computation, 
the key of the dictionary itself must be word1*-*-*-*word2 at the end of the computation.
'''

import numpy as np
import math
from collections import Counter

stopwords = set(["a","about","above","after","again","against","all","am","an","and","any","are","aren","'t","as","at","be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn","did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further","had","hadn","has","hasn","have","haven","having","he","he","'d","he","'ll","he","'s","her","here","here","hers","herself","him","himself","his","how","how","i","'m","'ve","if","in","into","is","isn","it","its","itself","let","'s","me","more","most","mustn","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan","she","she","'d","she","ll","she","should","shouldn","so","some","such","than","that","that","the","their","theirs","them","themselves","then","there","there","these","they","they","they","they","'re","they","this","those","through","to","too","under","until","up","very","was","wasn","we","we","we","we","we","'ve","were","weren","what","what","when","when","where","where","which","while","who","who","whom","why","why","with","won","would","wouldn","you","your","yours","yourself","yourselves"])

def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists) 
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters): 
        - frequency[y][x] = number of occurrences of bigram x in texts of class y,
          where x is in the format 'word1*-*-*-*word2'
    '''
    frequency = {'pos' : Counter(), 'neg' : Counter()}

    for text in train['pos']:
        for idx, token in enumerate(text):
            if idx == len(text) - 1:
                break
            bigram = token + '*-*-*-*' + text[idx + 1]
            frequency['pos'][bigram] += 1

    for text in train['neg']:
        for idx, token in enumerate(text):
            if idx == len(text) - 1:
                break
            bigram = token + '*-*-*-*' + text[idx + 1]
            frequency['neg'][bigram] += 1

    return frequency        


    #raise RuntimeError("You need to write this part!")

def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters): 
        - frequency[y][x] = number of occurrences of bigram x in texts of class y,
          where x is in the format 'word1*-*-*-*word2'
    stopwords (set of str):
        - Set of stopwords to be excluded

    Output:
    nonstop (dict of Counters): 
        - nonstop[y][x] = frequency of bigram x in texts of class y,
          but only if neither token in x is a stopword. x is in the format 'word1*-*-*-*word2'
    '''

    #nonstop = {'pos' : Counter(), 'neg' : Counter()}

    deletionspos = []
    deletionsneg = []

    for bigram in frequency['pos']:
        star1 = bigram.find('*')
        word1 = bigram[0:star1]
        word2 = bigram[star1+7:]
        if word1 in stopwords and word2 in stopwords:
            deletionspos.append(bigram)
    
    for bigram in frequency['neg']:
        star1 = bigram.find('*')
        word1 = bigram[0:star1]
        word2 = bigram[star1+7:]
        if word1 in stopwords and word2 in stopwords:
            deletionsneg.append(bigram)
    
    for stopword in deletionspos:
        del frequency['pos'][stopword]

    for stopword in deletionsneg:
        del frequency['neg'][stopword]
    
    nonstop = frequency
    return nonstop
    #raise RuntimeError("You need to write this part!")


def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of bigram x in y, where x is in the format 'word1*-*-*-*word2'
          and neither word1 nor word2 is a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of bigram x given y,
          where x is in the format 'word1*-*-*-*word2'
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary bigram given y

    Important: 
    Be careful that your vocabulary only counts bigrams that occurred at least once
    in the training data for class y.
    '''

    pos_count = 0
    neg_count = 0

    likelihood = {'pos' : Counter(), 'neg' : Counter()}

    for bigram in nonstop['pos']:
        if nonstop['pos'][bigram] > 0:
            likelihood['pos'][bigram] += nonstop['pos'][bigram]
            pos_count += nonstop['pos'][bigram]
    
    for bigram in nonstop['neg']:
        if nonstop['neg'][bigram] > 0:
            likelihood['neg'][bigram] += nonstop['neg'][bigram] 
            neg_count += nonstop['neg'][bigram]

    pos_factor = (pos_count) + smoothness * (len(likelihood['pos']) + 1)
    neg_factor = (neg_count) + smoothness * (len(likelihood['neg']) + 1)

    for bigram in likelihood['pos']:
        likelihood['pos'][bigram] += smoothness
        likelihood['pos'][bigram] /= pos_factor

    for bigram in likelihood['neg']:
        likelihood['neg'][bigram] += smoothness
        likelihood['neg'][bigram] /= neg_factor
    
    likelihood['pos']['OOV'] = smoothness/pos_factor
    likelihood['neg']['OOV'] = smoothness/neg_factor
    
    return likelihood
    #raise RuntimeError("You need to write this part!")


# turn list of tokens into a dictionary of bigrams with Counters
# with the format 'word1*-*-*-*word2'
def construct_bigrams(text):
    bigrams = []
    for i in range(len(text) - 1):
        bigram = text[i] + '*-*-*-*' + text[i+1]
        bigrams.append(bigram)
    return bigrams

def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of bigram x given y,
          where x is in the format 'word1*-*-*-*word2'
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''
    p_pos = math.log(prior)
    p_neg = math.log(1 - prior)

    text_bigrams = [] # list of bigram dictionaries for each text
    for text in texts:
        text_bigrams.append(construct_bigrams(text))

    hypotheses = []

    for text in text_bigrams:
        llsum_pos = 0
        llsum_neg = 0
        for bigram in text:
            word1, word2 = bigram.split('*-*-*-*')
            if word1 in stopwords and word2 in stopwords:
                continue

            if bigram in likelihood['pos']:
                llsum_pos += math.log(likelihood['pos'][bigram])
            else:
                llsum_pos += math.log(likelihood['pos']['OOV'])
                
            if bigram in likelihood['neg']:
                llsum_neg += math.log(likelihood['neg'][bigram])
            else:
                llsum_neg += math.log(likelihood['neg']['OOV'])

        pos_given = p_pos + llsum_pos
        neg_given = p_neg + llsum_neg
            
        if pos_given > neg_given:
            hypotheses.append('pos')
        elif pos_given < neg_given:
            hypotheses.append('neg')
        else:
            hypotheses.append('undecided')
  
    return hypotheses
    
    #raise RuntimeError("You need to write this part!")



def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''
    accuracies = np.zeros((len(priors), len(smoothnesses)))

    for idx, p in enumerate(priors):
        for idy, s in enumerate(smoothnesses):
            correct = 0
            likelihood = laplace_smoothing(nonstop, s)
            hypths = naive_bayes(texts, likelihood, p)
            for idz, label in enumerate(labels):
                if hypths[idz] == label:
                    correct += 1
            accuracies[idx][idy] = correct / len(labels)
        
    return accuracies
    #raise RuntimeError("You need to write this part!")
                          