def viterbi(test, train):
    '''
    Implementation for the viterbi tagger.
    input:  test data (list of sentences, no tags on the words)
            training data (list of sentences, with tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    initial = Counter()

    word_set = set()
    tag_set = set()

    initial_unknown_prob = 0
    transition_unknown_prob = 0
    emission_unknown_prob = 0

    # populate word set
    for sentence in train:
         for pair in sentence:
              word_set.add(pair[0])
    

    tag_freqs = Counter()
    # populate tag set
    for sentence in train:
         for pair in sentence:
              tag_freqs[pair[1]] += 1
              tag_set.add(pair[1])

    # indexes for words in matrix
    word_dict = {}
    for i, word in enumerate(list(word_set)):
        word_dict[word] = i
         
    # word_dict = {word : i for i in range(len(word_set))}

    # indexes for tags in matrix
    tag_dict = {}
    for j, tag in enumerate(list(tag_set)):
        tag_dict[tag] = j
    # tag_dict = {tag_set[j] : j for j in range(len(tag_set))}
                
    initial = Counter()

    transition = np.zeros((len(tag_set), len(tag_set)))

    emission = np.zeros((len(tag_set), len(word_set)))

    first_tag = True
    even = False
    tag1 = ""
    tag2 = ""
    
    # populate matrices
    for sentence in train:
         for pair in sentence:
              word = pair[0]
              tag = pair[1]

              # initial counts
              if first_tag:
                   initial[tag] += 1
                   first_tag = False

              # transition pairs
                   
              if not even:
                   tag1 = word
              elif even and tag1 in tag_dict:
                   even = False
                   tag2 = word
                   if tag2 in tag_dict:
                        transition[tag_dict[tag1]][tag_dict[tag2]] += 1
              else:
                   even = False
              
              # emission pairs
              if tag in tag_dict and word in word_dict:
                emission[tag_dict[tag]][word_dict[word]] += 1
              even = True
    
         even = False
         first_tag = True
     
    file_path = "initial.txt"
    with open(file_path, 'w') as file:
     for value in initial:
          file.write(str(value) + " ")
 
    file_path = "transition.txt"
    with open(file_path, 'w') as file:
     for row in transition:
          for value in row:
               file.write(str(value) + " ")
          file.write('\n')

    file_path = "emission.txt"
    with open(file_path, 'w') as file:
     for row in emission:
          for value in row:
               file.write(str(value) + " ")
          file.write('\n')
    

    # laplace smooth initial tags
    initial_unknown_prob = epsilon / (len(train) + (epsilon * (len(tag_set) + 1)))
    for tag in initial:
         initial[tag] += epsilon
         initial[tag] /= len(train) + (epsilon * (len(tag_set) + 1))
    

    # laplace smooth transition pairs
    row_sums = np.zeros(len(tag_set))
    row_sum = 0
    idx = 0

    transition_unknown_prob = 0.1 # adsadsasda
    for row in transition:
         for count in row:
              count += epsilon
              row_sum += count
         row_sums[idx] = row_sum
         row_sum = 0
         idx += 1
     
    idx = 0
    
    for row in transition:
         denominator = row_sums[idx] * (epsilon * (len(tag_set) + 1))
         row /= denominator
         idx += 1
    

    # laplace smooth emission pairs
    row_sums = np.zeros(len(word_set))
    idx = 0
    row_sum = 0 

    emission_unknown_prob = 0.1
    for row in emission:
         for count in row:
              count += epsilon
              row_sum += count
         row_sums[idx] = row_sum
         row_sum = 0
         idx += 1
     
    idx = 0
    
    for row in emission:
         denominator = row_sums[idx] * (epsilon * (len(word_set) + 1))
         row /= denominator
         idx += 1
    
    # log everything   
    for tag in initial:
         initial[tag] = math.log(initial[tag])
    
    transition = np.log(transition)

    emission = np.log(emission)

    file_path = "initial_laplace.txt"
    with open(file_path, 'w') as file:
     for value in initial:
          file.write(str(value) + " ")
 
    file_path = "transition_laplace.txt"
    with open(file_path, 'w') as file:
     for row in transition:
          for value in row:
               file.write(str(value) + " ")
          file.write('\n')

    file_path = "emission_laplace.txt"
    with open(file_path, 'w') as file:
     for row in emission:
          for value in row:
               file.write(str(value) + " ")
          file.write('\n')


    # construct the trellis

    # prob is pi times b
    # next is an array of tuples with (i, j, transition_prob)
    # where N is the index (i, j) of the next node and t is the transition probability

    def get_key_by_value(dict, value):
        for key, val in dict.items():
                if val == value:
                        return key

    class TrellNode:
         def __init__(self, prob=None, back=None):
                self.prob = prob
                self.back = back

         def get_prob(self):
                return self.prob
         
         def get_back(self):
                if self.back is None:
                        return None
                return self.back
         
    word_test_set = set()
    for sentence in test:
         for word in sentence:
              word_test_set.add(word)
    
    answer = []

    # go through test data
    for sentence in test:
         first = True
         trellis = np.empty((len(tag_set), len(sentence)), dtype = TrellNode)

         # set up trellis
         for col in range(len(sentence)):
              word = sentence[col]
              for row in range(len(tag_set)):
                emission_prob = 0
                tag_idx = row
                if word not in word_dict: # wrong
                     emission_prob = emission_unknown_prob
                else:
                     word_idx = word_dict[word]
                     emission_prob = emission[tag_idx][word_idx]
                initial_prob = initial[tag_idx]
                if first:
                        trellis[row][col] = TrellNode(initial_prob * emission_prob, None)
                else:
                        backptrs = [0] * len(tag_set)
                        for r in range(len(tag_set)): 
                                backptrs[r] = (r, col - 1, transition[row][r]) # trans prob might be wrong
                        trellis[row][col] = TrellNode(emission_prob, backptrs)
              first = False

         # find max prob node in last column of trellis
         answer_sentence = []
         max = trellis[0][len(sentence) - 1]
         max_row = 0
         for i in range(len(tag_set)):
                current = trellis[i][len(sentence) - 1]
                if current.get_prob() > max.get_prob():
                     max = current
                     max_row = i
         word = get_key_by_value(word_dict, len(sentence) - 1)
         tag = get_key_by_value(tag_dict, max_row)
         answer_sentence.append((word, tag))

         # create our answer sentence (word, tag)
         while max.get_back() is not None:
              backs = max.get_back()
              max_tuple = (0,0,0)
              for tuple_ptr in backs:
                   trans_prob = tuple_ptr[2]
                   if trans_prob >= max_tuple[2]:
                        max_tuple = tuple_ptr
              max = trellis[max_tuple[0]][max_tuple[1]]
              word = get_key_by_value(word_dict, max_tuple[1])
              tag = get_key_by_value(tag_dict, max_tuple[0])
              answer_sentence.append((word, tag))
        
         # answer_sentence.reverse()
         answer.append(answer_sentence)


    file_path = "debug.txt"
    with open(file_path, 'w') as file:
     for value in answer:
          file.write(str(value) + '\n')

    return answer 