from keras.preprocessing.text import one_hot
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, Bidirectional, Layer
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras import backend as K
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()


class self_model:
    def __init__(self,vs,em,maxlen = 100):
        self.seqModel = Sequential()
        self.embedding_layer = Embedding(vs, 100, weights=[em], input_length=maxlen , trainable=False)
        self.seqModel.add(self.embedding_layer)
        
    def simpleNN(self):
        self.seqModel.add(Flatten())
        self.seqModel.add(Dense(1, activation='sigmoid'))
        self.seqModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        print(self.seqModel.summary())
        return(self.seqModel)
    
    def LSTM(self):
        self.seqModel.add(LSTM(128))
        self.seqModel.add(Dense(1, activation='sigmoid'))
        self.seqModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        print(self.seqModel.summary())
        return(self.seqModel)
    
    def bi_LSTM(self, Att = False):
        if Att:
            self.seqModel.add(Bidirectional(LSTM(128,return_sequences=True)))
            self.seqModel.add(Attention(return_sequences=True))
            self.seqModel.add(LSTM(128))
        else:
            self.seqModel.add(Bidirectional(LSTM(128)))
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
# This Class is adopted from 
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