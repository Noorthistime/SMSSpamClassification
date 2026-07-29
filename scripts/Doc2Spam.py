#SMS Spam Classification [Code]
#Link for dataset download
#[http://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection , If can't get then get it from Kaggle.com]
#"Here is the dataset's link"

#Importing Libraries/Packages
import numpy as np 
import pandas as pd

import os
data_path = "../data/SMSSpamCollection" if os.path.exists("../data/SMSSpamCollection") else ("data/SMSSpamCollection" if os.path.exists("data/SMSSpamCollection") else "SMSSpamCollection")
dataset = pd.read_csv(data_path, sep='\t', names=['label', 'message'])
dataset
dataset.info()
dataset.describe()

dataset['label'] = dataset['label'].map({'ham':0 ,'spam':1})
dataset

#Visualizing data
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

#Count Plot for Spam vs Ham as imbalanced dataset
plt.figure(figsize=(8,8))
g = sns.countplot(x="label", data = dataset)
p = plt.title('Countplot for Spam vs Ham as imbalanced dataset')
p = plt.xlabel('Is the SMS Spam?')
p = plt.ylabel('Count')

#Handling imbalanced dataset using Oversampling
only_spam = dataset[dataset["label"] == 1]
only_spam 
print('Number of Spam SMS:', len(only_spam)) 
print('Number of Ham SMS:', len(dataset) - len(only_spam))  
count = int((dataset.shape[0] - only_spam.shape[0]) / only_spam.shape[0])
count
for i in range(0, count-1):
	dataset = pd.concat([dataset, only_spam])
dataset.shape


#Count Plot for Spam vs Ham as balanced dataset
plt.figure(figsize=(8,8))
g = sns.countplot(x="label", data = dataset)
p = plt.title('Countplot for Spam vs Ham as balanced dataset')
p = plt.xlabel('Is the SMS Spam?')
p = plt.ylabel('Count')

  #Creating a new feature word_count
dataset['word_count'] = dataset['message'].apply(lambda x: len(x.split()))
dataset
plt.figure(figsize=(12,6))

#(1,1)
plt.subplot(1,2,1)
g = sns.histplot(dataset[dataset["label"] == 0].word_count, kde = True)
p = plt.title('Distribution of word_count for Ham SMS')

#(1,2)
plt.subplot(1,2,2)
g = sns.histplot(dataset[dataset["label"] == 1].word_count, color = "red", kde = True)
p = plt.title('Distribution of word_count for Spam SMS')
plt.tight_layout()
plt.show()

#Creating new feature of containing currency symbols
def currency(data):
	currency_symbols = ['$','€','₹','¥','₺']
	for i in currency_symbols:
		if i in data:
			return 1
	return 0

dataset["contains_currency_symbols"] = dataset["message"].apply(currency)
dataset

#CountPlot for contains_currency_symbols
plt.figure(figsize=(8, 8))
g = sns.countplot(x='contains_currency_symbols',
                  data=dataset[dataset['contains_currency_symbols'] == 0],
                  hue='label')
plt.title('Countplot for containing currency symbols')
plt.xlabel('Does SMS contain any currency symbols?')
plt.ylabel('Count')
plt.legend(labels=["Ham", "Spam"], loc = 9)
plt.show()


#Creating new feature of containing Numbers
def number(data):
	for i in data:
		if ord(i) >= 48 and ord(i) <= 57:
			return 1
	return 0
dataset["contains_number"] = dataset['message'].apply(number)
dataset

#Countplot for containing numbers
plt.figsize=(8,8)
g = sns.countplot(x = 'contains_number', data = dataset, hue = "label")
p = plt.title("Countplot for containing number")
p = plt.xlabel('Does SMS contains any number?')
p = plt.ylabel('count')
p = plt.legend(["Ham", "Spam"], loc=9)

#Data Cleaning
import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

corpus = []
wnl = WordNetLemmatizer()
for sms in list(dataset.message):
	message = re.sub(pattern='[^a-zA-Z]', repl = ' ', string = sms) #Filtering out special characters and numbers
	message = message.lower()
	words = message.split() #Tokenizer
	filtered_words = [word for word in words if word not in set(stopwords.words('english'))]
	lemm_words = [wnl.lemmatize(word) for word in filtered_words]
	message = ' '.join(lemm_words)
	corpus.append(message)
corpus

#Creating the bag of words model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features = 500)
vectors = tfidf.fit_transform(corpus).toarray()
feature_names = tfidf.get_feature_names_out()

X = pd.DataFrame(vectors, columns =  feature_names)
y = dataset['label']

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
X_test

#Naive Bayes Model
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
cv = cross_val_score(mnb, X, y, scoring='f1', cv = 10)
print(round(cv.mean(),3))
print(round(cv.std(),3))

mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
cm

plt.figure(figsize=(8,8))
axis_labels = ["ham", "spam"]
g = sns.heatmap(data=cm, xticklabels=axis_labels, yticklabels=axis_labels, annot = True, fmt = 'g', cbar_kws = {"shrink":0.5}, cmap = "Blues")
p = plt.title("Confusion Matrix of Multinomial Naive Bayes Model")
p = plt.xlabel('Actual Labels')
p = plt.ylabel("Predicted values")

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
cv1 = cross_val_score(dt, X, y, scoring='f1', cv = 10)
print(round(cv1.mean(),3))
print(round(cv1.std(),3))

dt.fit(X_train, y_train)
y_pred1 = dt.predict(X_test)
print(classification_report(y_test, y_pred1))

cm = confusion_matrix(y_test, y_pred1)
cm

plt.figure(figsize=(8,8))
axis_labels = ["ham", "spam"]
g = sns.heatmap(data=cm, xticklabels=axis_labels, yticklabels=axis_labels, annot = True, fmt = 'g', cbar_kws = {"shrink":0.5}, cmap = "Blues")
p = plt.title("Confusion Matrix of Multinomial Naive Bayes Model")
p = plt.xlabel('Actual Labels')
p = plt.ylabel("Predicted values")


def predict_spam(sms):
	message = re.sub(pattern='[^a-zA-Z]', repl = ' ', string = sms) #Filtering out special characters and numbers
	message = message.lower()
	words = message.split() #Tokenizer
	filtered_words = [word for word in words if word not in set(stopwords.words('english'))]
	lemm_words = [wnl.lemmatize(word) for word in filtered_words]
	message = ' '.join(lemm_words)
	temp = tfidf.transform([message]).toarray()
	return mnb.predict(pd.DataFrame(temp, columns=feature_names))

#Prediction 1 - Lottery Ticket Exchange
sample_message = "IMPORTANT - You can be entitled up to $3160 from sis-sold PPI on a credit card or loan, Please check."
if predict_spam(sample_message):
	print('This is a SPAM message.')
else:
	print('This is a HAM(normal) message.')

#Prediction 2 - Casual Text Chat
sample_message = "Come to think of it, I have never got a spam message before."
if predict_spam(sample_message):
	print('This is a SPAM message.')
else:
	print('This is a HAM(normal) message.')

#Prediction 3 - Transaction Confirmation text Message
sample_message = "Sam, your rent payment for June 2022  has been recieved."
if predict_spam(sample_message):
	print('This is a SPAM message.')
else:
	print('This is a HAM(normal) message.')