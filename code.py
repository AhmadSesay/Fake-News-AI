import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')

print(stopwords.words('english'))

# Get the data from the sample
data = pd.read_csv("fake_or_real_news.csv")

# Store the author and text in a content variable
data_content = data['text']


# Turn true and false data into 1 for False and 0 for True

new_lable =  data['label'].apply(lambda x : 0 if x == "REAL" else 1)

# Seperate the the data and the label
X = data.drop(columns = 'label', axis= 1)
Y = new_lable

port_stem = PorterStemmer()


#Text editing method for for text data provided in data_contnet
def stemming(content):

    # removes everything that is not a letter from the text and replaes it with 
    # a space
    stemmed_content = re.sub('[^a-zA-Z]',' ', content)

    # the text is then set to lowercase and split
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()

    # Takes everywork in stemmed_content that is not  a stopword and reduced it 
    # to its root word and rejoins the words with a space

    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not
                        word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

data_content = data['text'].apply(stemming)

X = data_content.values
Y = new_lable.values

# converting text data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

# Splitting dataset into training and testsing data
X_train, X_test, y_train, y_test = train_test_split(X , Y, test_size=0.2 , stratify=Y, random_state=2)


# training model

model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy score for training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)

print('Accuracy score of the training data is : ', training_data_accuracy)

# Accuracy score for testing data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, y_test)

print('Accuracy score of the testing data is : ', testing_data_accuracy)



# predicing whether an article is fake for real
X_new = X_test[0]

prediction = model.predict(X_new)

print(prediction)

print("The article is Real") if prediction[0] == 0 else print("The article is Fake") 