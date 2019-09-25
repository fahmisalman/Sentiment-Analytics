from Model.Preprocessing import preprocessing
from Model.Model import GatedRecurrentUnit

import pandas as pd
import os


df = pd.read_csv(os.getcwd() + '/Dataset/Indonesian Sentiment Twitter Dataset Labeled.csv', delimiter='\t')

text = df['Tweet']
sentiment = df['sentimen']

x_train = []
y_train = []

for i in range(len(text)):
    x_train.append(preprocessing(text[i]))
    y_train.append(sentiment[i])

model = GatedRecurrentUnit()
model.fit(x_train, y_train)
model.save_model(model.model)
