for sentence in test:
          trellis = np.full((len(tag_set), len(sentence)), float('-inf'))
          answer_sentence = []

          # fill in first column
          for r in range(len(tag_set)):
               first_col_prob = 0

               if out_of_val((sentence[0], tag_lookup[r]), pair_set):
                    first_col_prob = initial[r] + observation["OOV"][tag_lookup[r]]
               else:
                    first_col_prob = initial[r] + observation[sentence[0]][tag_lookup[r]]
               
               trellis[r][0] = first_col_prob
          
          # fill in rest
          for w in range(1, len(sentence)):
               for t in range(len(tag_set)):
                    max_prob = float('-inf')
                    for r in range(len(tag_set)):
                         if out_of_val((tag_lookup[r], w), pair_set):
                              transition_prob = trellis[r][w - 1] + transition[t][r]
                              observation_prob = observation["OOV"][tag_lookup[t]]
                              current_prob = transition_prob + observation_prob
                         else:
                              transition_prob = trellis[r][w - 1] + transition[t][r]
                              observation_prob = observation[sentence[w]][tag_lookup[t]]
                              current_prob = transition_prob + observation_prob
                         
                         if current_prob > max_prob:
                              max_prob = current_prob
                              trellis[t][w] = max_prob
          
          # find max prob in last column
          back_max = float('-inf')
          back_idx = 0
          for r in range(len(tag_set)):
               if trellis[r][len(sentence)-1] > back_max:
                    back_max = trellis[r][len(sentence)-1]
                    back_idx = r
          
         # answer_sentence.append(('END', 'END'))
          answer_sentence.append((sentence[len(sentence)-1], tag_lookup[back_idx]))
          
          # backtrack based on highest probs in prev col
          for w in range(len(sentence)-2, -1, -1):
               back_max = float('-inf')
               back_idx = 0
               for r in range(len(tag_set)):
                    if trellis[r][w] >= back_max:
                         back_max = trellis[r][w]
                         back_idx = r
               answer_sentence.append((sentence[w], tag_lookup[back_idx]))
          
          #answer_sentence.append(('START', 'START'))