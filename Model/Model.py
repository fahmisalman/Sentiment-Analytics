from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.optimizers import Adam

import numpy as np
import joblib
import os


class GatedRecurrentUnit(object):

    def __init__(self, max_tokens=5000, embedding_size=8, num_words=10000, model=None, tokenizer=None):
        self.x_train = []
        self.y_train = []
        self.x_train_tokens = []
        self.list_label = []
        self.model = model
        self.num_words = num_words
        self.max_tokens = max_tokens
        self.embedding_size = embedding_size
        self.tokenizer = tokenizer
        self.summary = True
        self.verbose = 1
        self.epoch = 5
        self.validation_split = 0.1

    def one_hot_encoder(self, y):
        self.list_label = list(set(y))
        label = np.zeros([len(y), len(self.list_label)])
        for i in range(len(y)):
            label[i][self.list_label.index(y[i])] = 1
        return label

    def model_gru(self):

        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.num_words,
                                 output_dim=self.embedding_size,
                                 input_length=self.max_tokens,
                                 name='Embedding_Layer'))
        self.model.add(GRU(units=4))
        self.model.add(Dense(len(self.list_label),
                             activation='softmax',
                             name='Output_layer'))
        self.model.compile(loss='binary_crossentropy',
                           optimizer=Adam(lr=0.001),
                           metrics=['accuracy'])
        if self.summary:
            print(self.model.summary())

        self.model.fit(self.x_train_tokens,
                       self.y_train,
                       epochs=self.epoch,
                       validation_split=self.validation_split,
                       verbose=self.verbose)

    def text_to_seq(self, x):
        temp = self.tokenizer.texts_to_sequences([x])
        return pad_sequences(temp,
                             maxlen=self.max_tokens,
                             padding='pre',
                             truncating='pre')

    def fit(self, x_train, y_train, epoch=5, validation_split=0.1, verbose=1):

        self.x_train = x_train
        self.y_train = y_train

        self.epoch = epoch
        self.validation_split = validation_split
        self.verbose = verbose

        self.tokenizer = Tokenizer(num_words=self.num_words)
        self.tokenizer.fit_on_texts(self.x_train)

        self.x_train_tokens = self.tokenizer.texts_to_sequences(self.x_train)
        self.x_train_tokens = pad_sequences(self.x_train_tokens,
                                            maxlen=self.max_tokens,
                                            padding='pre',
                                            truncating='pre')

        # if type(self.y_train[0]) == str or type(self.y_train[0]) == int:
        self.y_train = np.array(self.one_hot_encoder(y_train))

        self.model_gru()

    def save_model(self, filename='model'):
        model_json = self.model.to_json()
        with open(os.path.join(os.getcwd(), 'Model/Output_model/{}.json'.format(filename)), 'w') as json_file:
            json_file.write(model_json)
        self.model.save_weights(os.path.join(os.getcwd(), 'Model/Output_model/{}.h5'.format(filename)))
        joblib.dump(self.tokenizer, os.path.join(os.getcwd(),
                                                 'Model/Output_model/{}_tokenizer.joblib'.format(filename)))
        joblib.dump(self.list_label, os.path.join(os.getcwd(),
                                                  'Model/Output_model/{}_list_label.joblib'.format(filename)))
        joblib.dump(self.max_tokens, os.path.join(os.getcwd(),
                                                  'Model/Output_model/{}_max_tokens.joblib'.format(filename)))

    def load_model(self, filename='model'):
        json_file = open(os.path.join(os.getcwd(), 'Model/Output_model/{}.json'.format(filename)), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(os.path.join(os.getcwd(), 'Model/Output_model/{}.h5'.format(filename)))
        self.tokenizer = joblib.load(os.path.join(os.getcwd(),
                                                  'Model/Output_model/{}_tokenizer.joblib'.format(filename)))
        self.list_label = joblib.load(os.path.join(os.getcwd(),
                                                   'Model/Output_model/{}_list_label.joblib'.format(filename)))
        self.max_tokens = joblib.load(os.path.join(os.getcwd(),
                                                   'Model/Output_model/{}_max_tokens.joblib'.format(filename)))
        self.model.compile(loss='binary_crossentropy',
                           optimizer=Adam(lr=0.001),
                           metrics=['accuracy'])

    def predict(self, x):
        return self.model.predict(x)

    def predict_classes(self, x):
        return self.model.predict_classes(x)

    def score(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)
