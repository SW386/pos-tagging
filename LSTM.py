#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 10:21:48 2021

@author: Shufan
"""

import sys
import numpy as np
from tqdm import tqdm
import nltk
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset

PAD_INDEX = 0
UNK_INDEX = 1
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
MAX_LENGTH = 256
BATCH_SIZE = 1
EMBEDDING_DIM = 256
HIDDEN_DIM = 256
OUTPUT_DIM = 12
N_LAYERS = 3
DROPOUT_RATE = 0.1
LR = 3e-4
N_EPOCHS = 5

UNIVERSAL_TAGS = [
    "VERB",
    "NOUN",
    "PRON",
    "ADJ",
    "ADV",
    "ADP",
    "CONJ",
    "DET",
    "NUM",
    "PRT",
    "X",
    ".",
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Tokenizer:
    
    def __init__(self):
        """
        Attributes
        ----------
        vocab : Dictionary
            Mapping of vocab word to integer representation
        tags : Dictionary
            Mapping of tag to integer representation

        Returns
        -------
        None.

        """
        
        self.vocab = {PAD_TOKEN : PAD_INDEX,
                      UNK_TOKEN : UNK_INDEX}
        self.tags = {w:i for i, w in enumerate(UNIVERSAL_TAGS)}
        
    def train(self, corpus):
        """
        Parameters
        ----------
        corpus : List of Lists
            Each list in corpus represents a sentence 
            Each list in corpus contains strings representing words

        Returns
        -------
        None.

        """
        counter = len(self.vocab)
        for sentence in tqdm(corpus, desc='Constructing Embeddings', file=sys.stdout):
            for word in sentence:
                if word not in self.vocab:
                    self.vocab[word] = counter
                    counter += 1
                    
        
    def tokenize(self, sentence):
        """
        Parameters
        ----------
        sentence : List
            List of words in the sentence

        Returns
        -------
        tokens : List
            List of integer representations of the sentence

        """
        tokens = []
        for word in sentence:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.vocab[UNK_TOKEN])
        return tokens
    
    def tag(self, tag):
        """
        Parameters
        ----------
        tag : String
            Specific tag

        Returns
        -------
        Integer
            Representation of a specific tag

        """
        if tag not in self.tags:
            return self.tags['X']
        return self.tags[tag]

class TaggedDataset(Dataset):
    
    def __init__(self, x, y, tokenizer, max_length):
        """
        Parameters
        ----------
        x : List of Lists
            Each list in x represents a sentence 
            Each list in x contains strings representing words
        y : List of Lists
            Each list contains tag
        tokenizer : Tokenizer
            Trained tokenizer only
        max_length : Integer
            Max length of sentence, if longer, truncate to be that length

        Returns
        -------
        None.

        """
        
        self.x = x
        self.y = y
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : Integer

        Returns
        -------
        ret : Dictionary
            Maps "ids" to a list representing a tokenized sentence
            Maps "label" to a list representing tokenized tags

        """
        
        tags = self.y[index]
        sentence = self.x[index]
        tokens = self.tokenizer.tokenize(sentence)
        if len(tokens) > self.max_length:
          tokens = tokens[:256]
          length = 256
        else:
          length = len(tokens)
        labels = []
        for i in range(length):
            tag = tags[i]
            label = self.tokenizer.tag(tag)
            labels.append(label)
        ret = {"ids" : tokens,
               "label" : labels}
        return ret

    def __len__(self):
        return len(self.x)


def collate(batch, pad_index):
    """
    Parameters
    ----------
    batch : List
        List of dictionaries that are indexed by TaggedDataset
    pad_index : Integer
        Pad index

    Returns
    -------
    batch : Dictionary
        Maps "ids" to a 2darray of padded sentences
        Maps "label" to a 2darray of labels
        

    """
    
    batch_ids = [torch.LongTensor(i['ids']) for i in batch]
    batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    batch_label = torch.LongTensor([i['label'] for i in batch])
    batch = {'ids': batch_ids, 'label': batch_label}
    return batch

class BiLSTMTagger(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout_rate, pad_index):
        """
        Parameters
        ----------
        vocab_size : Integer
            Length of vocab in Tokenizer
        embedding_dim : Integer
            Size of embedding input
        hidden_dim : Integer
            Size of hidden dimension of LSTM
        output_dim : Integer
            Size of output dimension of Linear Layer
        n_layers : Integer
            Number of layers in LSTM
        dropout_rate : Float
            Number between 0 and 1 indicating dropout strength
        pad_index : Integer
            padding index

        Returns
        -------
        None.

        """
        
        super(BiLSTMTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2*hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        
        
    def forward(self, ids):
        """
        Parameters
        ----------
        ids : Tensor
            1D tensor of ids representing a sentence
            Size of length N

        Returns
        -------
        out : Tensor
            2D tensor representing output of NN
            Dimensions of (N, output_dim)

        """
        
        embeddings = self.embedding(ids)
        embeddings = self.dropout(embeddings)
        out, _= self.lstm(embeddings)
        out = self.fc(out)
        return out

class Trainer:
    
    @staticmethod
    def train(dataloader, model, criterion, optimizer, device):
        """
        Parameters
        ----------
        dataloader : torch.utils.data.Dataloader
            Dataloader for TaggedDataset
        model : BiLSTMTagger
            LSTM network to train
        criterion : torch.nn.CrossEntropyLoss
            Loss Function for Training
        optimizer : torch.nn.Adam
            Optimizer for Training
        device : String
            'cpu' or 'cuda'

        Returns
        -------
        epoch_losses : List
            List of losses
        epoch_accs : List
            List of accuracies

        """
        model.train()
        epoch_losses = []
        epoch_accs = []

        for batch in tqdm(dataloader, desc='Training', file=sys.stdout):
            ids = batch['ids'].to(device)
            label = batch['label'].to(device)
            label = label.squeeze(dim=0)
            prediction = model(ids).squeeze(dim=0)
            loss = criterion(prediction, label)
            accuracy = Trainer.get_accuracy(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())

        return epoch_losses, epoch_accs
    
    @staticmethod
    def evaluate(dataloader, model, criterion, device):
        """
        Parameters
        ----------
        dataloader : torch.utils.data.Dataloader
            Dataloader for TaggedDataset
        model : BiLSTMTagger
            LSTM network to train
        criterion : torch.nn.CrossEntropyLoss
            Loss Function for Training
        device : String
            'cpu' or 'cuda'

        Returns
        -------
        epoch_losses : List
            List of losses
        epoch_accs : List
            List of accuracies

        """
        model.eval()
        epoch_losses = []
        epoch_accs = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating', file=sys.stdout):
                ids = batch['ids'].to(device)
                label = batch['label'].to(device)
                label = label.squeeze(dim=0)
                prediction = model(ids).squeeze(dim=0)
                loss = criterion(prediction, label)
                accuracy = Trainer.get_accuracy(prediction, label)
                epoch_losses.append(loss.item())
                epoch_accs.append(accuracy.item())

        return epoch_losses, epoch_accs
    
    @staticmethod
    def get_accuracy(prediction, label):
        """
        Parameters
        ----------
        prediction : Tensor
            Tensor of size (N, Num_Tags)
        label : Tensor
            Tensor of size N

        Returns
        -------
        Float
            Accuracy percentage

        """
        size, _ = prediction.shape
        predicted_classes = prediction.argmax(dim=-1)
        correct_predictions = predicted_classes.eq(label).sum()
        return correct_predictions / size

    @staticmethod
    def count_parameters(model):
        """
        Parameters
        ----------
        model : BiLSTMTagger
            LSTM network

        Returns
        -------
        Integer
            Number of parameters in model

        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    
    @staticmethod
    def predict_tags(text, model, tokenizer, device):
        """
        Parameters
        ----------
        text : List
            List of strings representing a sentence
        model : BiLSTMTagger
            Model to use for tagging
        tokenizer : Tokenizer
            Trained tokenizer only
        device : String
            'cpu' or 'cuda'

        Returns
        -------
        predicted_class : List
            List of integers representing the tag
        predicted_probability : List
            List of floats representing the probability of tags

        """
        tokens = tokenizer.tokenize(text)
        tensor = torch.LongTensor(tokens).unsqueeze(dim=0).to(device)
        prediction = model(tensor).squeeze(dim=0)
        probability = torch.softmax(prediction, dim=-1)
        predicted_class = probability.argmax(dim=-1).to('cpu').numpy()
        predicted_probability = []
        for i in range(len(tokens)):
            c = predicted_class[i]
            predicted_probability.append(probability[i][c].item())
        predicted_probability = np.array(predicted_probability)
        return predicted_class, predicted_probability

def initialize_weights(m):
    """
    Parameters
    ----------
    m : BiLSTMTagger
        LSTM network

    Returns
    -------
    None.

    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
               
def build_dataset(loc=""):
    """
    Parameters
    ----------
    loc : String, optional
        Location of dataset. The default uses the Brown Corpus

    Returns
    -------
    x : List of Lists
        Each list in x represents a sentence 
        Each list in x contains strings representing words
    y : List of Lists
        Each list contains tag
    """
    
    if loc == "":
        data = nltk.corpus.brown.tagged_sents(tagset="universal")
    else: 
        f = open(loc, 'r')
        data = json.load(f)
    
    x = []
    y = []
    
    for sentence in data:
        words = []
        tags = []
        for word, tag in sentence:
            words.append(word)
            tags.append(tag)
        x.append(words)
        y.append(tags)
        
    return x, y