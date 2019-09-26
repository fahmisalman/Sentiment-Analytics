from Model.Preprocessing import preprocessing
from Model.Model import GatedRecurrentUnit

import pandas as pd
import joblib
import os


df = pd.read_csv(os.getcwd() + '/Dataset/Indonesian Sentiment Twitter Dataset Labeled.csv', delimiter='\t')

text = df['Tweet']
sentiment = df['sentimen']

x_train = []
y_train = []

for i in range(len(text)):
    x_train.append(preprocessing(text[i]))
    y_train.append(sentiment[i])

model = GatedRecurrentUnit(embedding_size=8, max_tokens=10)
model.fit(x_train, y_train, epoch=1)
model.save_model()
# joblib.dump(model, os.path.join(os.getcwd(), 'Model/Output_model/{}.joblib'.format('model')))
joblib.dump(model.tokenizer, 'Model/Output_model/model.joblib')
