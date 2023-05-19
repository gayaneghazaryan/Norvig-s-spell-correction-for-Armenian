import fastDamerauLevenshtein as fdlt 
import pandas as pd
import numpy as np
import re


class Spell_checker:
    def __init__(self, corpus):
        self.word_counts = self.__calculate_word_freq(self.__preprocess_text(corpus))
        self.proximity_scores = self.__calculate_proximity_matrix()

    def __preprocess_text(self,text):
        cleaned_text = re.sub(r'[^\w\s]', '', text)
        cleaned_text = re.sub(r'[^\u0531-\u0556\u0561-\u0587\s]', '', cleaned_text)
        cleaned_text = re.sub(r'\n', ' ', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text.strip())

        tokens = cleaned_text.split()

        return tokens

    def __calculate_proximity_matrix(self):
        keyboard_map = {
        'է': [(0,0)], 'թ': [(0,1)], 'փ': [(0,2)], 'ձ': [(0,3)], 'ջ': [(0,4)], 'և': [(0,6)], 'ր': [(0,7)], 'չ': [(0,8)], 'ճ': [(0,9)], 'ժ': [(0,11)],
        'ք': [(1,0)], 'ո': [(1,1)], 'ե': [(1,2)], 'ռ': [(1,3)], 'տ': [(1,4)], 'ը': [(1,5)], 'ւ': [(0,5),(1,6)], 'ի': [(1,7)], 'օ': [(1,8)], 'պ': [(1,9)], 'խ': [(1,10)], 'ծ': [(1,11)], 'շ': [(1,12)],
        'ա': [(2,0)], 'ս': [(2,1)], 'դ': [(2,2)], 'ֆ': [(2,3)], 'գ': [(2,4)], 'հ': [(2,5)], 'յ': [(2,6)], 'կ': [(2,7)], 'լ': [(2,8)], 
        'զ': [(3,0)], 'ղ': [(3,1)], 'ց': [(3,2)], 'վ': [(3,3)], 'բ': [(3,4)], 'ն': [(3,5)], 'մ': [(3,6)]
        }
        letters = list(keyboard_map.keys())
        proximity_scores = np.zeros((39,39))
        for i in range(len(letters)):
            for j in range(i+1, len(letters)):
                char1 = letters[i]
                char2 = letters[j]
                positions1 = keyboard_map[char1]
                positions2 = keyboard_map[char2]
                if char1 == char2:
                    proximity_scores[ord(char1) - ord('ա')][ord(char2) - ord('ա')] = 0
                else:
                    min_distance = float('inf')
                    for pos1 in positions1:
                        for pos2 in positions2:
                            
                            distance = abs(pos1[0]-pos2[0]) + abs(pos1[1] - pos2[1])
                            min_distance = min(min_distance, distance)
                    proximity_scores[ord(char1)-ord('ա')][ord(char2)-ord('ա')] = min_distance 
        return proximity_scores

    

    def __calculate_word_freq(self,text):
        general_word_counts = pd.Series(text).value_counts(normalize=True).to_dict()
        word_counts = dict()

        for word, count in general_word_counts.items():
            lowercase_word = word.lower()
            if lowercase_word in general_word_counts.keys():
                if lowercase_word not in word_counts.keys():
                    word_counts[lowercase_word]  = count
                else:
                    word_counts[lowercase_word] += count
            else:
                word_counts[word] = count
        
        return word_counts
    
    def __calculate_proximity_score(self,misspelled_word, correct_word):
        score = 0 
        for a, b in zip(misspelled_word, correct_word):
            index_a = ord(a.lower()) - ord('ա')
            index_b = ord(b.lower()) - ord('ա')
            score += self.proximity_scores[index_a][index_b]

        if score == 0:
            return 0
        else:
            return 1/(1+score) #sigmoid
        
    def __calculate_letter_group_score(self,misspelled_word, correct_word):
        letter_groups = {
        'բ': ['պ', 'փ'],
        'պ': ['բ', 'փ'],
        'փ': ['բ', 'պ'],
        'գ': ['կ', 'ք'],
        'կ': ['գ', 'ք'],
        'ք': ['գ', 'կ'],
        'դ': ['տ', 'թ'],
        'տ': ['դ', 'թ'],
        'թ': ['դ', 'տ'],
        'ձ': ['ծ', 'ց'],
        'ծ': ['ձ', 'ց'],
        'ց': ['ծ', 'ձ'],
        'ջ': ['ճ', 'չ'],
        'ճ': ['ջ', 'չ'],
        'չ': ['ճ', 'ջ'],
        'ղ': ['խ'],
        'խ': ['ղ'],
        'զ': ['ս'],
        'ս': ['զ'],
        'վ': ['ֆ'],
        'ֆ': ['վ'],
        'ր': ['ռ'],
        'ռ': ['ր'],
        'է': ['ե'],
        'ե': ['է'],
        'օ': ['ո'],
        'ո': ['օ']
    }
        score = 0
        for a, b in zip(misspelled_word, correct_word):
            if a in letter_groups and b in letter_groups[a]:
                score += 0.5

        return score
    
    def find_correction(self, misspelled_word):
        if misspelled_word in self.word_counts.keys():
            return 'Վրիպակ չկա'
        else:
        
            word_distances = []

            max_transpositions = 1
            max_additions = 1
            max_deletions = 1
            max_substitutions = 2

            addition_weight = 0.1
            deletion_weight = 0.5
            transposition_weight = 0.2
            freq_weight = 10

            distance_threshold = 0.5


            for word, count in self.word_counts.items():
                distance = fdlt.damerauLevenshtein(misspelled_word, word)

                transpositions = sum(1 for a, b in zip(misspelled_word, word) if a != b and misspelled_word.index(a) == word.index(b))
                additions = len(word) - len(misspelled_word)
                deletions = len(misspelled_word) - len(word)
                substitutions = sum(1 for i, j in zip(misspelled_word, word) if i != j)
                if (
                    transpositions <= max_transpositions
                    and additions <= max_additions
                    and deletions <= max_deletions
                    and substitutions <= max_substitutions
                    and distance > distance_threshold 
                ):
                    score = distance + freq_weight * count
                    if substitutions > 0: 
                        score = score + self.__calculate_proximity_score(misspelled_word, word) + self.__calculate_letter_group_score(misspelled_word, word)
                    else:
                        score += transposition_weight * transpositions + deletion_weight * deletions + addition_weight * additions
                    word_distances.append((word, round(score,2)))
                    
            
            top_words = sorted(word_distances, key= lambda x: x[1], reverse = True)
            if len(top_words) == 0:
                return 'Շտկման տարբերակ չկա'
            else:
                return top_words 
        