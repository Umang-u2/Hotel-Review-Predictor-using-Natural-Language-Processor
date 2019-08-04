#importing libraries
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing dataset
yelp = pd.read_csv('yelp.csv')
yelp.head()
#Making a new column of the length of text data
yelp['text length']= yelp['text'].apply(len)

#Data Visuaalization and analysis
g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'text length',bins=30)
sns.boxplot(x='stars',y='text length',data=yelp)
sns.countplot(x='stars',data=yelp)
stars = yelp.groupby('stars').mean()
sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)

#Only taking the datas which have 5 star and 1 star ratings for making things easier
yelp_class = yelp[(yelp['stars']==1) | (yelp['stars']==5)]
yelp_class

#Declaring X and Y
X = yelp_class['text']
y= yelp_class['stars']

#Performing count vectorizer of bag of words on the model for tokenizing the text
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

#Splitting the data into training and testng data
from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
X_test

#Performing Naive Bayes Classification
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)
predictions = nb.predict(X_test)

#Performing predictive analysis
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

#Creating a pipeline for various preprocessing
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
pipe = Pipeline([('bow',CountVectorizer()),
                 ('tfidf',TfidfTransformer()),
                 ('model',MultinomialNB())])

#Performing the prediction again with pipeline and TFIFDF
X = yelp_class['text']
y= yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
pipe.fit(X_train,y_train)
predictions = pipe.predict(X_test)
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))
