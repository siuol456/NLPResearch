# Basic Packages
import pandas as pd
import numpy as np
import re
import nltk

# Modeling
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, Bidirectional, Layer
from keras.layers import Embedding
from keras.activations import softmax
from keras import backend as K

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

## Function adopted from 
#Preprovessing function to remove tags, html, special characters,and etc
class pre_pro:
    def __init__(self):
        self.TAG_RE = re.compile(r'<[^>]+>')
    def preprocess_text(self,inputSentence):
        # Removing html tags
        sentence = self.TAG_RE.sub('', inputSentence)

        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)

        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)

        return sentence

class self_model:
    def __init__(self,vs,em,maxlen = 100,d=100):
        self.maxlen = maxlen
        self.seqModel = Sequential()
        self.embedding_layer = Embedding(vs, d, weights=[em], input_length=maxlen , trainable=False)
        self.seqModel.add(self.embedding_layer)
        
    def simpleNN(self):
        self.seqModel.add(Flatten())
        self.seqModel.add(Dense(1, activation='sigmoid'))
        self.seqModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        print(self.seqModel.summary())
        return(self.seqModel)
    
    def LSTM(self, soft = False, results = 1):
        self.seqModel.add(LSTM(128))
        if soft:
            self.seqModel.add(Dense(results, activation='softmax'))
            self.seqModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        else:
            self.seqModel.add(Dense(1, activation='sigmoid'))
            self.seqModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        print(self.seqModel.summary())
        return(self.seqModel)
    
    def bi_LSTM(self, Att = False,soft = False, results = 1):
        if Att:
            self.seqModel.add(Bidirectional(LSTM(128,return_sequences=True)))
            self.seqModel.add(Attention(return_sequences=True))
            self.seqModel.add(LSTM(128))
        else:
            self.seqModel.add(Bidirectional(LSTM(128)))
        if soft:
            self.seqModel.add(Dense(results, activation='softmax'))
            self.seqModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        else:
            self.seqModel.add(Dense(1, activation='sigmoid'))
            self.seqModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        print(self.seqModel.summary())
        return(self.seqModel)
    
    def show_performance_plot(history):
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10, 5))
        #show the model accuracy
        ax1.plot(history.history['acc'],label='train')
        ax1.plot(history.history['val_acc'],label='test')

        ax1.set(title='model accuracy',xlabel='epoch',ylabel='accuracy')
        ax1.legend(loc='upper left')
        #show model loss
        ax2.plot(history.history['loss'],label='train')
        ax2.plot(history.history['val_loss'],label='test')

        ax2.set(title='model loss',xlabel='epoch',ylabel='loss')
        ax2.legend(loc='upper right')
        fig.tight_layout()
        return(fig)

    
# The Following Class for attention layer is adopted from 
# https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm
    
class Attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention,self).__init__()
        
    def build(self, input_shape):

        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                                   initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                                   initializer="zeros")

        super(Attention,self).build(input_shape)

    def call(self, x):

        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)
class performance_evplot:
    def __init__(self,h):
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10, 5))
        #show the model accuracy
        ax1.plot(h.history['acc'],label='train')
        ax1.plot(h.history['val_acc'],label='test')

        ax1.set(title='model accuracy',xlabel='epoch',ylabel='accuracy')
        ax1.legend(loc='upper left')
        #show model loss
        ax2.plot(h.history['loss'],label='train')
        ax2.plot(h.history['val_loss'],label='test')

        ax2.set(title='model loss',xlabel='epoch',ylabel='loss')
        ax2.legend(loc='upper right')
        fig.tight_layout()