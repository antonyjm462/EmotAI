import json
import keras
import numpy as np
import sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, LSTM
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, concatenate
from tensorflow.keras.models import Model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

def token():
    file = os.path.join(THIS_FOLDER, 'static/tokenizer.pickle')
    infile = open(file,'rb')
    tokenizer = pickle.load(infile)
    infile.close()
    return tokenizer

def encoder():
    labels = ['Happy','Sad','Anger','Disgust','Surprise','Fear','Bad']
    encoder = LabelBinarizer()
    encoder.fit(labels)
    return encoder

def model():
    input_length = 428
    input_dim = 203169
    num_classes = 7
    embedding_dim = 500
    lstm_units = 128
    lstm_dropout = 0.1
    recurrent_dropout = 0.1
    filters=64
    kernel_size=3
    input_layer = Input(shape=(input_length,))
    output_layer = Embedding(
      input_dim=input_dim,
      output_dim=embedding_dim,
      input_shape=(input_length,)
    )(input_layer)
    output_layer = Bidirectional(
    LSTM(lstm_units, return_sequences=True,
         dropout=lstm_dropout, recurrent_dropout=recurrent_dropout)
    )(output_layer)
    output_layer = Conv1D(filters, kernel_size=kernel_size, padding='valid',
                        kernel_initializer='glorot_uniform')(output_layer)

    avg_pool = GlobalAveragePooling1D()(output_layer)
    max_pool = GlobalMaxPooling1D()(output_layer)
    output_layer = concatenate([avg_pool, max_pool])

    output_layer = Dense(num_classes, activation='softmax')(output_layer)
    model = Model(input_layer, output_layer)
    file = os.path.join(THIS_FOLDER, 'static/model.h5')
    model.load_weights(file)
    return model


class Emotion:



    def __init__(self):
        self.model = model()
        self.tokenizer = token()
        self.encoder = encoder()

    def test(self,text):
        labels = ['Happy','Sad','Anger','Disgust','Surprise','Fear','Bad']
        tokenized = self.tokenizer.texts_to_sequences([text])
        pad_data = pad_sequences(tokenized,428)
        pred = self.model.predict(pad_data)
        print(pred,flush=True)
        emotion = labels[np.argmax(pred)]
        confidence = pred[0][np.argmax(pred)] * 100
        return (emotion,confidence)