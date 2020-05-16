from models import simple_model
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
from process import load_data_and_labels_one,load_data_and_labels_two
from utils import *
import pickle
def POS():

    X_train, y_train = load_data_and_labels_one("conll-2003/eng.train")
    X_valid, y_valid= load_data_and_labels_one("conll-2003/eng.testa")
    X_teste, y_teste= load_data_and_labels_one("conll-2003/eng.testb")


    all_words, all_tags = build_uniques(X_train + X_valid + X_teste, y_train)

    MAX_LEN = max([len(x) for x in X_train + X_valid + X_teste])
    NUM_WORDS = len(all_words) + 2
    NUM_TAGS = len(all_tags) + 1

    X_train, y_train = parser_arrays(MAX_LEN,X_train, y_train, all_words, all_tags)
    X_valid, y_valid = parser_arrays(MAX_LEN,X_valid, y_valid, all_words, all_tags)
    X_teste, y_teste = parser_arrays(MAX_LEN,X_teste, y_teste, all_words, all_tags)

    with open('encoding/X_teste_POS.pkl', 'wb') as f:
        pickle.dump(X_teste, f)
    with open('encoding/y_teste_POS.pkl', 'wb') as f:
        pickle.dump(y_teste, f)
    with open('encoding/all_words_POS.pkl', 'wb') as f:
        pickle.dump(all_words, f)
    with open('encoding/all_tags_POS.pkl', 'wb') as f:
        pickle.dump(all_tags, f)
    with open('encoding/MAX_LEN_POS.pkl', 'wb') as f:
        pickle.dump(MAX_LEN, f)

    model=simple_model(NUM_WORDS,MAX_LEN,NUM_TAGS)
    filename = 'checkpoints/model_POS.h5'
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model.fit(X_train,
                        y_train,
                        batch_size=64,
                        epochs=5,  
                        validation_data = [X_valid, y_valid],  callbacks=[checkpoint],
                        verbose=1)

    print("Saved model to disk")

def PAR():

    X_train1, y_train1, y_train2 = load_data_and_labels_two("conll-2003/eng.train")
    X_valid1, y_valid1, y_valid2= load_data_and_labels_two("conll-2003/eng.testa")
    X_teste1, y_teste1, y_teste2= load_data_and_labels_two("conll-2003/eng.testb")


    all_words, all_tags = build_uniques(X_train1 + X_valid1 + X_teste1, y_train1)

    MAX_LEN = max([len(x) for x in X_train1 + X_valid1 + X_teste1])
    NUM_WORDS = len(all_words) + 2
    NUM_TAGS = len(all_tags) + 1

    X_train, y_train = parser_arrays(MAX_LEN,X_train1, y_train1, all_words, all_tags)
    X_valid, y_valid = parser_arrays(MAX_LEN,X_valid1, y_valid1, all_words, all_tags)
    X_teste, y_teste = parser_arrays(MAX_LEN,X_teste1, y_teste1, all_words, all_tags)


    with open('encoding/X_teste_PAR.pkl', 'wb') as f:
        pickle.dump(X_teste, f)
    with open('encoding/y_teste_PAR.pkl', 'wb') as f:
        pickle.dump(y_teste, f)
    with open('encoding/all_words_PAR.pkl', 'wb') as f:
        pickle.dump(all_words, f)
    with open('encoding/all_tags_PAR.pkl', 'wb') as f:
        pickle.dump(all_tags, f)
    with open('encoding/MAX_LEN_PAR.pkl', 'wb') as f:
        pickle.dump(MAX_LEN, f)

    model=simple_model(NUM_WORDS,MAX_LEN,NUM_TAGS)
    filename = 'checkpoints/model_PAR.h5'
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model.fit(X_train,
                        y_train,
                        batch_size=64,
                        epochs=5,  
                        validation_data = [X_valid, y_valid],  callbacks=[checkpoint],
                        verbose=1)

    print("Saved model to disk")
    

def NER():

    X_train1, y_train1, y_train2 = load_data_and_labels_two("conll-2003/eng.train")
    X_valid1, y_valid1, y_valid2= load_data_and_labels_two("conll-2003/eng.testa")
    X_teste1, y_teste1, y_teste2= load_data_and_labels_two("conll-2003/eng.testb")


    all_words, all_tags = build_uniques(X_train1 + X_valid1 + X_teste1, y_train2)

    MAX_LEN = max([len(x) for x in X_train1 + X_valid1 + X_teste1])
    NUM_WORDS = len(all_words) + 2
    NUM_TAGS = len(all_tags) + 1

    X_train, y_train = parser_arrays(MAX_LEN,X_train1, y_train2, all_words, all_tags)
    X_valid, y_valid = parser_arrays(MAX_LEN,X_valid1, y_valid2, all_words, all_tags)
    X_teste, y_teste = parser_arrays(MAX_LEN,X_teste1, y_teste2, all_words, all_tags)


    with open('encoding/X_teste_NER.pkl', 'wb') as f:
        pickle.dump(X_teste, f)
    with open('encoding/y_teste_NER.pkl', 'wb') as f:
        pickle.dump(y_teste, f)
    with open('encoding/all_words_NER.pkl', 'wb') as f:
        pickle.dump(all_words, f)
    with open('encoding/all_tags_NER.pkl', 'wb') as f:
        pickle.dump(all_tags, f)
    with open('encoding/MAX_LEN_NER.pkl', 'wb') as f:
        pickle.dump(MAX_LEN, f)

    model=simple_model(NUM_WORDS,MAX_LEN,NUM_TAGS)
    filename = 'checkpoints/model_NER.h5'
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model.fit(X_train,
                        y_train,
                        batch_size=64,
                        epochs=5,  
                        validation_data = [X_valid, y_valid],  callbacks=[checkpoint],
                        verbose=1)

    print("Saved model to disk")

def main():
    print("Hi! Please wait till the models are trained and ready to be in use..")
    print("Training the POS Model..")
    POS()
    print("Training the PAR Model..")
    PAR()
    print("Training the NER Model..")
    NER()

main()