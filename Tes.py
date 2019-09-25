# import pandas as pd
# import numpy as np
# import re
# import os
#
# from tensorflow.python.keras.utils import to_categorical
# from tensorflow.python.keras.preprocessing.text import Tokenizer
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences
#
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, GRU, Embedding
# from tensorflow.python.keras.optimizers import Adam
#
#
# def preprocessing(text):
#     text = re.sub('\n', '. ', text)
#     text = text.lower()
#     text = re.sub(r'[^a-z]', ' ', text)
#     return text
#
#
# def label_cleansing(text):
# #     text = re.sub('Dataset', '', text)
# #     text = re.sub('.txt', '', text)
# #     text = re.sub('/', '', text)
# #     text = re.sub(r'[^a-z]', '', text)
#     return text
#
#
# def one_hot_encoder(y):
#     list_label = list(set(y))
#     label = np.zeros([len(y), len(list_label)])
#     for i in range(len(y)):
#         label[i][list_label.index(y[i])] = 1
#     return label
#
#
# def fit(x_train_tokens, y_train, max_tokens=5000, embedding_size=8, num_words=10000, summary=False):
#
#     model = Sequential()
#     model.add(Embedding(input_dim=num_words,
#                         output_dim=embedding_size,
#                         input_length=max_tokens,
#                         name='layer_embedding'))
#     # model.add(GRU(units=16, return_sequences=True))
#     # model.add(GRU(units=8, return_sequences=True))
#     model.add(GRU(units=64))
#     model.add(Dense(3, activation='softmax'))
#     optimizer = Adam(lr=1e-3)
#     model.compile(loss='binary_crossentropy',
#                   optimizer=optimizer,
#                   metrics=['accuracy'])
#     y_train = np.array(y_train)
#     if summary:
#         print(model.summary())
#     model.fit(x_train_tokens, y_train, epochs=5, validation_split=0.1)
#     return model
#
#
# df = pd.read_csv(os.getcwd() + '/Dataset/Indonesian Sentiment Twitter Dataset Labeled.csv', delimiter='\t')
#
# text = df['Tweet']
# sentiment = df['sentimen']
#
# x_train = []
# y_train = []
#
# for i in range(len(text)):
#     x_train.append(preprocessing(text[i]))
#     y_train.append(label_cleansing(sentiment[i]))
#
# num_words = 10000
# max_tokens = 500
# embedding_size = 256
#
# tokenizer = Tokenizer(num_words=num_words)
# tokenizer.fit_on_texts(x_train)
#
# x_train_tokens = tokenizer.texts_to_sequences(x_train)
# x_train_tokens = pad_sequences(x_train_tokens, maxlen=max_tokens, padding='pre', truncating='pre')
# y_train = one_hot_encoder(y_train)
#
# model = fit(x_train_tokens, y_train, max_tokens=max_tokens,
#             embedding_size=embedding_size,
#             num_words=num_words,
#             summary=True)
#
# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

from Model import Train