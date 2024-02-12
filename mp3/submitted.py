'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np
import utils

# define your epsilon for laplace smoothing here
epsilon = 1e-5

def baseline(test, train):
    '''
    Implementation for the baseline tagger.
    input:  test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
            training data (list of sentences, with tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    # remove duplicates out of test data
    test_set = set()
    for sentence in test:
          for word in sentence:
                test_set.add(word)

    frequencies = {word: Counter() for word in test_set}
    tag_freqs = Counter()

    # {word: {tag1: 0, tag2: 2, tagn: n}}
    
    for sentence in train:
         for pair in sentence:
              word = pair[0]
              tag = pair[1]
              tag_freqs[tag] += 1
              if word in test_set:
                   frequencies[word][tag] += 1

    answer = []
    tag_max_freq = tag_freqs.most_common(1)[0][0]

    for sentence in test:
         new_sentence = []
         for word in sentence:
              if frequencies[word]:
                   new_sentence.append((word, max(frequencies[word], key=frequencies[word].get)))
              else:
                   new_sentence.append((word, tag_max_freq))
         answer.append(new_sentence)

    return answer

    # raise NotImplementedError("You need to write this part!")


def viterbi(test, train):
     '''
     Implementation for the viterbi tagger.
     input:  test data (list of sentences, no tags on the words)
               training data (list of sentences, with tags on the words)
     output: list of sentences with tags on the words
               E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
     '''

     word_set = set()
     tag_set = set()
     pair_set = set()
    
     for sentence in train:
         for pair in sentence:
              word_set.add(pair[0])
              tag_set.add(pair[1])
              pair_set.add(pair)
     
     tag_dict = {tag : idx for idx, tag in enumerate(list(tag_set))}

     tag_lookup = {value: key for key, value in tag_dict.items()}

     tags_list = [tag_lookup[i] for i in range(len(tag_lookup))]

     initial = np.zeros(len(tag_set))
     tag_counts = np.zeros(len(tag_set))

     transition = np.zeros((len(tag_set), len(tag_set)))

     observation = {word : Counter() for word in word_set}

     # calculate initial tag counts
     for i in range(len(train)):
          for j in range(len(train[i])):
               pair = train[i][j]
               tag = pair[1]
               tag_idx = tag_dict[tag]
               if j == 0:
                    initial[tag_idx] += 1
               tag_counts[tag_idx] += 1      
     
     # calculate transition tag counts
     # P(Y_t = j | Y_t-1 = i) = P(Y_t = j | Y_t-1 = i) / P(Y_t-1 = i)
     # rows are prob, cols are given (P(rows | cols) = P(rows and cols) / P(cols))
     for i in range(len(train)):
          for j in range(len(train[i])): # cant compare 0th index to -1
               if j == len(train[i])-1:
                    break
               pair1 = train[i][j]
               pair2 = train[i][j+1]
               tag1 = pair1[1]
               tag2 = pair2[1]
               tag_idx1 = tag_dict[tag1]
               tag_idx2 = tag_dict[tag2]
               transition[tag_idx2][tag_idx1] += 1
     
     # calculate observation tag counts
               
     unique_words = Counter()
     word_set2 = set()
               
     for i in range(len(train)):
          for j in range(len(train[i])):
               pair = train[i][j]
               word = pair[0]
               tag = pair[1]
               observation[word][tag] += 1

     
     # ---------------------------------------------------------------- calc laplace probs
               
     # initial laplace probs
               
     # k_count = 0
     # for count in tag_counts:
     #      k_count += epsilon + count
     
     total_tags = 0
     for sentence in train:
          for word in sentence:
               total_tags += 1

     for i in range(len(initial)):
               initial[i] += epsilon
               initial[i] /= (len(train) + epsilon * len(tag_set))

     for i in range(len(initial)):
          initial[i] = math.log(initial[i])
     
     # transition laplace probs
     # P(t2 | t1)
     # column sum = # of any t2 given t1
          
     counts_tag = Counter()
     for word in observation:
          for tag in observation[word]:
               counts_tag[tag] += observation[word][tag]
     
     col_sum = np.zeros(len(transition))
     for c in range(len(transition)):
          for r in range(len(transition[c])):
               col_sum[c] += transition[r][c]

     for r in range(len(transition)):
          for c in range(len(transition[c])):
                    transition[r][c] += epsilon
                    transition[r][c] /= col_sum[r] + epsilon
     
     transition = np.log(transition)


     # emission laplace probs
     
     unique_words = Counter()
     for tag in tag_set:
          word_set2 = set()
          for i in range(len(train)):
               for j in range(len(train[i])):
                    pair = train[i][j]
                    word = pair[0]
                    tag = pair[1]
                    if word not in word_set2:
                         word_set2.add(word)
                         unique_words[tag] += 1
     
     for word in observation:
          for tag in observation[word]:
                    observation[word][tag] += (epsilon)
                    denominator = counts_tag[tag] + (epsilon) * (unique_words[tag] + 1)
                    observation[word][tag] /= denominator
     
     observation["OOV"] = Counter()

     for tag in tags_list:
          observation["OOV"][tag] = (epsilon) / (counts_tag[tag] + ((epsilon) * (unique_words[tag] + 1)))
     
     for word in observation:
          for tag in observation[word]:
               observation[word][tag] = math.log(observation[word][tag])
     

     answer = []
     # calculate viterbi

     for ids, sentence in enumerate(test):
          answer_sentence = []
          viterbi = np.zeros((len(tag_set), len(sentence)))
          backpointer = np.zeros((len(tag_set), len(sentence)))

          for t, tag in enumerate(tags_list):
               obs_prob = observation["OOV"][tag]


               if not out_of_val((sentence[0], tag), pair_set):
                         obs_prob = observation[sentence[0]][tag]

               viterbi[t][0] = initial[t] + obs_prob
               backpointer[t][0] = 0
          
          for time_step, word in enumerate(sentence):
               if time_step == 0:
                    continue
               for state in range(len(tag_set)):
                    tag = tag_lookup[state]

                    obs_prob = observation["OOV"][tag]
                    if not out_of_val((sentence[time_step], tag), pair_set):
                         obs_prob = observation[word][tag]
                    
                    # probbilities are negative
                    maxvit = float('-inf')
                    maxvit_idx = 0
                    for sprime in range(len(tag_set)):
                         if viterbi[sprime][time_step - 1] + transition[state][sprime] + obs_prob > maxvit:
                              maxvit = viterbi[sprime][time_step - 1] + transition[state][sprime] + obs_prob
                              maxvit_idx = sprime

                    viterbi[state][time_step] = maxvit
                    backpointer[state][time_step] = maxvit_idx

          bestpathprob = viterbi[len(tag_set) - 1, len(sentence) - 1]
          bestpathpointer = len(tag_set) - 1 # row idx

          for s in range(len(tag_set)):
               if viterbi[s][len(sentence) - 1] > bestpathprob:
                    bestpathprob = viterbi[s][len(sentence) - 1]
                    bestpathpointer = s

          for w_idx in range(len(sentence) - 1, -1, -1):
               answer_sentence.append((sentence[w_idx], tag_lookup[bestpathpointer]))
               bestpathpointer = int(backpointer[bestpathpointer][w_idx])

     
          # reverse since we added in opposite order
          answer_sentence.reverse()
          answer.append(answer_sentence)
               
     return answer
               
     
     
     
def out_of_val(pair, pair_set):
     if pair in pair_set:
               return False
     return True

     

     



     

    

        




# raise NotImplementedError("You need to write this part!")


def viterbi_ec(test, train):
     '''
     Implementation for the improved viterbi tagger.
     input:  test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
               training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
     output: list of sentences, each sentence is a list of (word,tag) pairs.
               E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
     '''
     word_set = set()
     tag_set = set()
     pair_set = set()
     hapax = Counter()
    
     for sentence in train:
         for pair in sentence:
              word_set.add(pair[0])
              tag_set.add(pair[1])
              pair_set.add(pair)
              hapax[pair[0]] += 1

     hapax_set = set()

     for word in hapax:
          if hapax[word] == 1:
               hapax_set.add(word)
     
     hapax_tag_counts = {word : Counter() for word in hapax_set}
     
     tag_dict = {tag : idx for idx, tag in enumerate(list(tag_set))}

     tag_lookup = {value: key for key, value in tag_dict.items()}

     tags_list = [tag_lookup[i] for i in range(len(tag_lookup))]

     initial = np.zeros(len(tag_set))
     tag_counts = np.zeros(len(tag_set))

     transition = np.zeros((len(tag_set), len(tag_set)))

     observation = {word : Counter() for word in word_set}

     # calculate initial tag counts
     for i in range(len(train)):
          for j in range(len(train[i])):
               pair = train[i][j]
               word = pair[0]
               tag = pair[1]
               tag_idx = tag_dict[tag]
               if j == 0:
                    initial[tag_idx] += 1
               if word in hapax_set:
                    hapax_tag_counts[word][tag] += 1
               tag_counts[tag_idx] += 1      
     
     # calculate transition tag counts
     # P(Y_t = j | Y_t-1 = i) = P(Y_t = j | Y_t-1 = i) / P(Y_t-1 = i)
     # rows are prob, cols are given (P(rows | cols) = P(rows and cols) / P(cols))
     for i in range(len(train)):
          for j in range(len(train[i])): # cant compare 0th index to -1
               if j == len(train[i])-1:
                    break
               pair1 = train[i][j]
               pair2 = train[i][j+1]
               tag1 = pair1[1]
               tag2 = pair2[1]
               tag_idx1 = tag_dict[tag1]
               tag_idx2 = tag_dict[tag2]
               transition[tag_idx2][tag_idx1] += 1
     
     # calculate observation tag counts
               
     unique_words = Counter()
     word_set2 = set()
               
     for i in range(len(train)):
          for j in range(len(train[i])):
               pair = train[i][j]
               word = pair[0]
               tag = pair[1]
               observation[word][tag] += 1

     
     # ---------------------------------------------------------------- calc laplace probs
               
     # initial laplace probs
               
     # k_count = 0
     # for count in tag_counts:
     #      k_count += epsilon + count
     
     total_tags = 0
     for sentence in train:
          for word in sentence:
               total_tags += 1

     for i in range(len(initial)):
               initial[i] += epsilon
               initial[i] /= (len(train) + epsilon * len(tag_set))

     for i in range(len(initial)):
          initial[i] = math.log(initial[i])
     
     # transition laplace probs
     # P(t2 | t1)
     # column sum = # of any t2 given t1
          
     counts_tag = Counter()
     for word in observation:
          for tag in observation[word]:
               counts_tag[tag] += observation[word][tag]
     
     col_sum = np.zeros(len(transition))
     for c in range(len(transition)):
          for r in range(len(transition[c])):
               col_sum[c] += transition[r][c]

     for r in range(len(transition)):
          for c in range(len(transition[c])):
                    transition[r][c] += epsilon
                    transition[r][c] /= col_sum[r] + epsilon
     
     transition = np.log(transition)


     # emission laplace probs
     
     unique_words = Counter()
     for tag in tag_set:
          word_set2 = set()
          for i in range(len(train)):
               for j in range(len(train[i])):
                    pair = train[i][j]
                    word = pair[0]
                    tag = pair[1]
                    if word not in word_set2:
                         word_set2.add(word)
                         unique_words[tag] += 1

     # hapax
               
     hapax_tag_probs = {tag : 0 for tag in tag_set}

     for word in hapax_tag_counts:
          for tag in hapax_tag_counts[word]:
               hapax_tag_probs[tag] += 1

     
     # uniquetagshapax = 0
     # for word in hapax_tag_counts:
     #      for tag in hapax_tag_counts:
     #           if hapax_tag_counts[word][tag] == 1:
     #                uniquetagshapax += 1


     hapax_word_count = sum(hapax_tag_probs.values()) 

     # laplace smooth hapax probs
     for tag in hapax_tag_probs:
          hapax_tag_probs[tag] += epsilon
          hapax_tag_probs[tag] /= hapax_word_count + (1 * (len(tag_set) + 1))
     
     for word in observation:
          for tag in observation[word]:
                    if tag in hapax_tag_probs:
                         observation[word][tag] += (epsilon * hapax_tag_probs[tag])
                         denominator = counts_tag[tag] + (epsilon * hapax_tag_probs[tag])* (unique_words[tag] + 1)
                    else:
                         observation[word][tag] += epsilon 
                         denominator = counts_tag[tag] + (epsilon)* (unique_words[tag] + 1)
                    observation[word][tag] /= denominator

     
     observation["OOV"] = Counter()
     for tag in tags_list:
          if tag in hapax_tag_probs:
               observation["OOV"][tag] = (epsilon * hapax_tag_probs[tag]) / (counts_tag[tag] + ((epsilon * hapax_tag_probs[tag]) * (unique_words[tag] + 1)))
          else:
               observation["OOV"][tag] = (epsilon) / (counts_tag[tag] + ((epsilon) * (unique_words[tag] + 1)))
     
     for word in observation:
          for tag in observation[word]:
               observation[word][tag] = math.log(observation[word][tag])
     

     answer = []
     # calculate viterbi

     for ids, sentence in enumerate(test):
          answer_sentence = []
          viterbi = np.zeros((len(tag_set), len(sentence)))
          backpointer = np.zeros((len(tag_set), len(sentence)))

          for t, tag in enumerate(tags_list):
               obs_prob = observation["OOV"][tag]


               if not out_of_val((sentence[0], tag), pair_set):
                         obs_prob = observation[sentence[0]][tag]

               viterbi[t][0] = initial[t] + obs_prob
               backpointer[t][0] = 0
          
          for time_step, word in enumerate(sentence):
               if time_step == 0:
                    continue
               for state in range(len(tag_set)):
                    tag = tag_lookup[state]

                    obs_prob = observation["OOV"][tag]
                    if not out_of_val((sentence[time_step], tag), pair_set):
                         obs_prob = observation[word][tag]
                    
                    # probbilities are negative
                    maxvit = float('-inf')
                    maxvit_idx = 0
                    for sprime in range(len(tag_set)):
                         if viterbi[sprime][time_step - 1] + transition[state][sprime] + obs_prob > maxvit:
                              maxvit = viterbi[sprime][time_step - 1] + transition[state][sprime] + obs_prob
                              maxvit_idx = sprime

                    viterbi[state][time_step] = maxvit
                    backpointer[state][time_step] = maxvit_idx

          bestpathprob = viterbi[len(tag_set) - 1, len(sentence) - 1]
          bestpathpointer = len(tag_set) - 1 # row idx

          for s in range(len(tag_set)):
               if viterbi[s][len(sentence) - 1] > bestpathprob:
                    bestpathprob = viterbi[s][len(sentence) - 1]
                    bestpathpointer = s

          for w_idx in range(len(sentence) - 1, -1, -1):
               answer_sentence.append((sentence[w_idx], tag_lookup[bestpathpointer]))
               bestpathpointer = int(backpointer[bestpathpointer][w_idx])

     
          # reverse since we added in opposite order
          answer_sentence.reverse()
          answer.append(answer_sentence)
               
     return answer
    



