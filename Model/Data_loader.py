import pandas as pd
import numpy as np
import os


df = pd.read_csv('dataset/tweets.csv')
text = df['text']
sentiment = df['airline_sentiment']