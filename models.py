
import numpy as np
from glob import glob
from utils import *
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

def simple_model(NUM_WORDS,MAX_LEN,NUM_TAGS):
    model = Sequential()
    model.add(Embedding(input_dim=NUM_WORDS, output_dim=MAX_LEN,input_length=MAX_LEN))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(units=MAX_LEN,return_sequences=True,recurrent_dropout=0.1)))
    model.add(TimeDistributed(Dense(units=NUM_TAGS)))
    model.add(Activation('softmax'))
    model.compile(optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
    return model
