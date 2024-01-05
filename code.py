import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

data = pd.read_csv("fake_or_real_news.csv")

data['fake'] = data['label'].apply(lambda x : 0 if x == "REAL" else 1)

X , Y = data['text'], data['fake']

X_train, X_test, y_train, y_test = train_test_split(X , Y, test_size=0.2)

vectorizer 



