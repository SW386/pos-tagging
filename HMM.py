#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 19:26:36 2021

@author: Shufan
"""

import numpy as np
import nltk
from collections import defaultdict
from tqdm import tqdm
import sys
import json
import random

class HMMTagger:
    
    def __init__(self):
        """
        Attributes
        ----------
        vocab : List
            List of all Vocab Words 
        vocab_mapping : Dict
            Mapping of Vocab to an Integer
        vocab_probs : NP Array
            Probability of seeing a Vocab Word
        tags : List
            List of all Tags
        tag_mapping : Dict
            Mapping of Tags to an Integer
        tag_probs : NP Array
            Probability of seeing a Tag
        transition_matrix : 2D Numpy Array of size NxN
            Probability of transitioning from tag i to tag j.
            N is the length of the phrase list
        observation_matrix : 2D Numpy Array of size MxN
            Probability of seeing word i with tag j
            N is the length of the phrase list
            M is the length of the word list + 1 for unknown words
        vocab_matrix: 2D Numpy Array of size NxN
            Probability of transitiong from word i to word j.
            M is the length of the word list
        tag_starting : 1D Numpy Array of size N
            Probability each sentence starts with tag i
        vocab_starting : 1D Numpy Array of size M
            Probability each sentence starts with word i

        Returns
        -------
        None.

        """
        
        self.vocab = []
        self.vocab_mapping = defaultdict(int)
        self.vocab_probs = None
        self.tags = []
        self.tag_mapping = defaultdict(int)
        self.tag_probs = None
        
        self.transition_matrix = None
        self.observation_matrix = None
        self.vocab_matrix = None
        self.tag_starting = None
        self.vocab_starting = None
        
        self.M = 0
        self.N = 0
        
    def train(self, corpus):
        """

        Parameters
        ----------
        corpus : List of Lists
            Each list in corpus represents a sentence
            Each list contains tuples, where the first item is a word and the second is POS tag
        
        Returns
        -------
        None.

        """
        #Count all unique words and tags
        tag_count = defaultdict(int)
        vocab_count = defaultdict(int)
        for sentence in tqdm(corpus, desc='Constructing Embeddings', file=sys.stdout):
            for word, tag in sentence:
                vocab_count[word] += 1
                tag_count[tag] += 1
        #Get all vocab and tags
        self.vocab = list(vocab_count.keys())
        self.tags = list(tag_count.keys())
        #Generate mappings for vocab and tags
        M = len(vocab_count) + 1
        N = len(tag_count)
        self.M = M
        self.N = N
        self.tag_mapping = dict(zip(self.tags, [i for i in range(N)]))
        for i in range(1, M):
            word = self.vocab[i-1]
            self.vocab_mapping[word] = i
        #Generate probability of each tag
        total_tags = sum(list(self.tag_mapping.values()))
        self.tag_probs = np.zeros((N))
        for tag, count in tag_count.items():
            i = self.tag_mapping[tag]
            self.tag_probs[i] = count/total_tags
        #create probability matrices
        transition_matrix_count = np.zeros((N,N))
        observation_matrix_count = np.zeros((M,N))
        vocab_matrix_count = np.zeros((M,M))
        starting_tag_count = np.zeros((N))
        starting_vocab_count = np.zeros((M))
        #for transition matrices, map i->j to be the transition
        #for observation matrics, map i->j to be the observation
        for sentence in tqdm(corpus, desc='Training HMM', file=sys.stdout):
            prev_tag = None
            prev_word = None
            for word, tag in sentence:
                #count instances where the previous tag transitions to current tag
                if prev_tag != None:
                    prev_index = self.tag_mapping[prev_tag]
                    curr_index = self.tag_mapping[tag]
                    transition_matrix_count[prev_index][curr_index] += 1
                else:
                    #if the instance is the first, we count 
                    tag_index = self.tag_mapping[tag]
                    starting_tag_count[tag_index] += 1
                #count instances where the previous word transitions to current word
                if prev_word != None:
                    prev_index = self.vocab_mapping[prev_word]
                    curr_index = self.vocab_mapping[word]
                    vocab_matrix_count[prev_index][curr_index] += 1
                else:
                    word_index = self.vocab_mapping[word]
                    starting_vocab_count[word_index] += 1
                prev_tag = tag
                #count the observation of a word having a specific tag
                word_index = self.vocab_mapping[word]
                phrase_index = self.tag_mapping[tag]
                observation_matrix_count[word_index][phrase_index] += 1
        #convert counts to probability
        transition_matrix_probs = transition_matrix_count
        observation_matrix_probs = observation_matrix_count
        vocab_matrix_probs = vocab_matrix_count
        #use plus one smoothing
        for i in tqdm(range(N), desc='Computing Transition Probs', file=sys.stdout):
            total_transitions = np.sum(transition_matrix_probs[i,:]) + 1
            transition_matrix_probs[i,:] += 1 
            transition_matrix_probs /= total_transitions
        for i in tqdm(range(1, M), desc='Computing Observation Probs', file=sys.stdout):
            total_observations = np.sum(observation_matrix_probs[i,:]) + 1
            observation_matrix_probs[i,:] += 1
            observation_matrix_probs[i,:] /= total_observations
        for i in tqdm(range(1, M), desc='Computing Vocab Probs', file=sys.stdout):
            total_vocab = np.sum(vocab_matrix_probs[i, :]) + 1
            vocab_matrix_probs[i,:] += 1
            vocab_matrix_probs[i,:] /= total_vocab
        
    
        total_starting = np.sum(starting_tag_count)
        starting_prob =  starting_tag_count/ total_starting
        
        total_vocab_starting = np.sum(starting_vocab_count)
        vocab_starting = starting_vocab_count / total_vocab_starting
        
        self.transition_matrix = transition_matrix_probs
        self.observation_matrix = observation_matrix_probs
        self.vocab_matrix = vocab_matrix_probs
        self.tag_starting = starting_prob
        self.vocab_starting = vocab_starting
    
    def viterbi(self, obs):
        """
        Parameters
        ----------
        obs : List
            List of integers representing observations, size T

        Returns
        -------
        states : List
            List of integers representing the tag to be decoded
        """
        
        T = len(obs)
        N = self.N
        #create dp and backtracking arrays
        dp = np.zeros((T, N))
        backtracking = np.zeros((T-1, N))
        #add a tiny to prevent log 0
        tiny = np.finfo(0.).tiny
        PLog = np.log(np.array(self.tag_starting) + tiny)
        ALog = np.log(self.transition_matrix + tiny) 
        BLog = np.log(self.observation_matrix + tiny)
        #for the observation at 0, add the probability of seeing it at any given state to starting probs
        if obs[0] == 0:
            dp[0, :] = PLog + self.tag_probs
        else:
            dp[0, :] = PLog + BLog[obs[0], :] 
        for t in range(1, T):
            for n in range(N):
                #a single row of n gives transition from n->next in ALog
                #since T is the first part of the shape, we take the prev
                temp = ALog[n, :]+ dp[t-1, :]
                #for an unknown word, we take the probability a tag appears
                if obs[t] == 0:
                    dp[t, n] = np.max(temp) + self.tag_probs[n]
                else:
                    dp[t, n] = np.max(temp) + BLog[obs[t], n]
                backtracking[t-1, n] = np.argmax(temp) #store index for backtracking
        states = [0 for t in range(T)]
        last_opt = np.argmax(dp[-1,:])
        states[-1] = int(last_opt)
        for n in range(T-2, -1, -1):
            states[n] = int(backtracking[n][states[n+1]]) #at each state n, obtain the index that led to the n+1 state
    
        return states
    
    def predict(self, obs):
        """
        Parameters
        ----------
        obs : List
            List of integers representing observations, size T

        Returns
        -------
        states : List
            List of integers representing the tag to be decoded
        """
        states = self.viterbi(obs)
        pred = [self.tags[i] for i in states]
        return pred
    
    def test(self, corpus):
        """
        Parameters
        ----------
        corpus : List of Lists
            Each list in corpus represents a sentence
            Each list contains tuples, where the first item is a word and the second is POS tag
        
        Returns
        -------
        accuracy : Float
            percentage correct in a corpus
        """
        
        correct = 0
        total = 0
        
        for case in corpus:
            
            obs = []
            sentence = []
            truth = []
            
            for word, tag in case:
                
                sentence.append(word)
                truth.append(tag)
                obs.append(self.vocab_mapping[word])
            
            pred = self.predict(obs)
            t = len(pred)
            for i in range(t):
                total += 1
                if pred[i] == truth[i]:
                    correct += 1
                    
        return correct/total
                
            
    
    def generate(self, start, length=10, percent_x=0.4):
        """
        Parameters
        ----------
        start : List of Strings
            start words to generate data
        length : Integer
            length of synthetic data to generate

        Returns
        -------
        synthetic_data:
            list of tuples, where the first item is a word and the second is POS tag
        """
        
        ret = start.copy()
        
        if len(ret) == 0:
            vocab = self.vocab
            probs = self.vocab_starting[1:]
            word = np.random.choice(vocab, p=probs)
            ret.append(word)
    
        while len(ret) < length:
            prev = ret[-1]
            prev_idx = self.vocab_mapping[prev]
            vocab = self.vocab
            probs = self.vocab_matrix[prev_idx][1:]
            prob_sum = np.sum(probs)
            probs = probs/prob_sum #remove effects of +1 smoothing
            word = np.random.choice(vocab, p=probs)
            word = str(word)
            ret.append(word)
        
        ints = []
        for word in ret:
            ints.append(self.vocab_mapping[word])
        preds = self.predict(ints)
        
        #regenerate data if too many unknown tags are present
        num_x = 0
        for pred in preds:
            if pred == "X":
                num_x += 1
        if num_x > percent_x * length:
            return self.generate(start, length=length, percent_x=percent_x)

        synthetic_data = []
        for i in range(length):
            word = str(ret[i])
            tag = preds[i]
            synthetic_data.append((word,tag))
        return synthetic_data
            
        
        
if __name__ == "__main__":
    
    
    data = list(nltk.corpus.brown.tagged_sents(tagset="universal"))
    #f = open('synthetic_data.json', 'r')
    #data = json.load(f)
    #data = list(data.values())
    
    random.shuffle(data)
    N = len(data)
    print(N)
    frac = 0.2
    
    train_size = int(N*frac)

    
    train = data[:train_size]
    test = data[train_size:]
    
    model = HMMTagger()
    model.train(train)
    
    correct = 0
    total = 0
    
    for index, case in tqdm(enumerate(test), desc='Testing', file=sys.stdout):
        
        obs = []
        sentence = []
        truth = []
        for tup in case:
            #check if we have seen the word before, if not, we use 0
            #since word_mapping is a defaultdict, we automatically get 0 if unknown
            sentence.append(tup[0])
            obs.append(model.vocab_mapping[tup[0]])
            truth.append(tup[1])
        
        pred = model.predict(obs)
        
        #Calculate Accuracy
        N = len(pred)
        for i in range(N):
            if pred[i] == truth[i]:
                correct += 1
        total += N

    
    print(correct/total)
    
    
        
                
        
        
        