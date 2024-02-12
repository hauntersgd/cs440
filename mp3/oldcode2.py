for ids, sentence in enumerate(test):
          answer_sentence = []
          viterbi = np.zeros((len(tag_set), len(sentence)))
          backpointer = np.zeros((len(tag_set), len(sentence)))

          for t, tag in enumerate(tags_list):
               obs_prob = observation["OOV"][tag]


               if sentence[0] in word_set:
                         obs_prob = observation[sentence[0]][tag]

               viterbi[t][0] = initial[t] + obs_prob
               backpointer[t][0] = 0
          
          for time_step, word in enumerate(sentence):
               if time_step == 0:
                    continue
               for state in range(len(tag_set)):
                    tag = tag_lookup[state]

                    obs_prob = obs_prob = observation["OOV"][tag]
                    if word in word_set:
                         obs_prob = observation[word][tag]
                    
                    # probbilities are negative
                    maxvit = float('-inf')
                    maxvit_idx = 0
                    for sprime in range(len(tag_set)):
                         if viterbi[sprime][time_step - 1] + transition[sprime][state] + obs_prob > maxvit:
                              maxvit = viterbi[sprime][time_step - 1] + transition[sprime][state] + obs_prob
                              maxvit_idx = sprime

                    viterbi[state][time_step] = maxvit
                    backpointer[state][time_step] = maxvit_idx

          bestpathprob = viterbi[len(tag_set) - 1, len(sentence) - 1]
          bestpathpointer = len(tag_set) - 1 # row idx

          for s in range(len(tag_set)):
               if viterbi[s][len(sentence) - 1] > bestpathprob:
                    bestpathprob = viterbi[s][len(sentence) - 1]
                    bestpathpointer = s

          #answer_sentence.append(('END', 'END'))
          for w_idx in range(len(sentence) - 1, -1, -1):
               answer_sentence.append((sentence[w_idx], tag_lookup[bestpathpointer]))
               bestpathpointer = int(backpointer[bestpathpointer][w_idx])
          #answer_sentence.append(('START', 'START'))