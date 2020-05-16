import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from keras.models import Sequential
from keras.layers import (LSTM, 
                          Embedding,
                          Dense,
                          TimeDistributed,GlobalMaxPooling1D,
                          Dropout,Conv1D,
                          Bidirectional,Input,Activation)

from seqeval.metrics import (precision_score,
                             recall_score,
                             f1_score,
                             classification_report,
                             accuracy_score)

import gc

def plot(history, arr):
    FONTS = 18
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    for idx in range(2):
        ax[idx].grid(True)
        
        ax[idx].plot(history.history[arr[idx][0]], dashes=[5, 3], lw = 4)
        ax[idx].plot(history.history[arr[idx][1]], dashes=[6, 2], lw = 3)
        ax[idx].legend([arr[idx][0], arr[idx][1]], fontsize=FONTS)
            
        ax[idx].set_xlabel('Epoch ',fontsize=FONTS)
        ax[idx].set_ylabel('Metric',fontsize=FONTS)
        ax[idx].set_title(arr[idx][0] + ' X ' + arr[idx][1],fontsize=FONTS)
        
def metrics(pred_tag, true_tag):

    print(classification_report(pred_tag, true_tag))
    print('=' * 25)
    print("Precision: \t", precision_score(pred_tag, true_tag))
    print("Recall: \t", recall_score(pred_tag, true_tag))
    print("F1: \t\t", f1_score(pred_tag, true_tag))

def build_uniques(arr_x, arr_y):
    
    tmp_x, tmp_y = [], []
    
    for idx in arr_x: 
        for x in idx: 
            tmp_x.append(x)
  
    for idx in arr_y: 
        for x in idx:
            tmp_y.append(x)
    return list(set(tmp_x)), list(set(tmp_y))

## Helper Functions
def word2idx(all_words): 

    tmp = {value: idx + 2 for idx, value in enumerate(all_words)}
    tmp["UNK"] = 1 
    tmp["PAD"] = 0
    
    return tmp

def tag2idx(all_tags):
    
    tmp = {value: idx + 1 for idx, value in enumerate(all_tags)}
    tmp["PAD"] = 0 
    
    return tmp

def idx2word(word2idx):
    
    return {idx: value for value, idx in word2idx.items()}

def idx2tag(tag2idx):
    
    return {idx: value for value, idx in tag2idx.items()}



def parser_arrays(MAX_LEN,x_train, y_train, all_words, all_tags):
    
    obj_word2idx = word2idx(all_words)
    obj_tag2idx = tag2idx(all_tags)
    
    __X = [[obj_word2idx[x] for x in value] for value in x_train] 
    __y = [[obj_tag2idx[x] for x in value] for value in y_train]

    #Terceira Parte
    X_pad = pad_sequences(maxlen=MAX_LEN, sequences=__X, padding="post", value=0)
    y_pad = pad_sequences(maxlen=MAX_LEN, sequences=__y, padding="post", value=0)
    
    return  X_pad, np.array([to_categorical(idx, num_classes=len(all_tags) + 1) for idx in y_pad])

def parser2categorical(pred, y_true, all_tags):
    
    k = tag2idx(all_tags)
    parser_idx = idx2tag(k)

    pred_tag = [[parser_idx[idx] for idx in row] for row in pred]
    y_true_tag = [[parser_idx[idx] for idx in row] for row in y_true] 
    
    return pred_tag, y_true_tag
def build_matrix(path, ALLWORDS, DIM):
    
    embeddings_index = {}
    
    with open(path) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((NUM_WORDS, DIM))
    for word, i in ALLWORDS.items():
        if i >= NUM_WORDS:  
            continue
        try:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: 
                embedding_matrix[i] = embedding_vector
        except:
            embedding_matrix[i] = embeddings_index.get('UNK')
            
    del embeddings_index
    gc.collect()
    
    return embedding_matrix