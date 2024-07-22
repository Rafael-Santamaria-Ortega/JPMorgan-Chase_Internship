import pandas as pd
import os

def read_spam():
    category = 'spam'
    directory = './enron1/spam'
    return read_category(category, directory)

def read_ham():
    category = 'ham'
    directory = './enron1/ham'
    return read_category(category, directory)

def read_category(category, directory):
    emails = []
    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(directory, filename), 'r') as fp:
            try:
                content = fp.read()
                emails.append({'name': filename, 'content': content, 'category': category})
            except:
                print(f'skipped {filename}')
    return emails

ham = read_ham()
spam = read_spam()

df_ham=pd.DataFrame.from_records(ham)
df_spam=pd.DataFrame.from_records(spam)
df=pd.concat([df_ham, df_spam], ignore_index=True)

import re

def preprocessor(e):
    x=re.sub(r'[^a-zA-Z]',' ',e)
    e_reg=x.lower()
    return e_reg

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# The CountVectorizer converts a text sample into a vector (think of it as an array of floats).
# Each entry in the vector corresponds to a single word and the value is the number of times the word appeared.
# Instantiate a CountVectorizer. Make sure to include the preprocessor you previously wrote in the constructor.
vectorizer=CountVectorizer(preprocessor=preprocessor,analyzer='word')
y=df['category'].map({'spam':1,'ham':0})

# Use train_test_split to split the dataset into a train dataset and a test dataset.
# The machine learning model learns from the train dataset.
# Then the trained model is tested on the test dataset to see if it actually learned anything.
# If it just memorized for example, then it would have a low accuracy on the test dataset and a high accuracy on the train dataset.
X_train,X_test,y_train,y_test=train_test_split(df['content'],y,test_size=0.2,train_size=0.8,shuffle=True,random_state=42)


# Use the vectorizer to transform the existing dataset into a form in which the model can learn from.
# Remember that simple machine learning models operate on numbers, which the CountVectorizer conveniently helped us do.
X_train_trans=vectorizer.fit_transform(X_train)
X_test_trans=vectorizer.transform(X_test)

# Use the LogisticRegression model to fit to the train dataset.
# You may remember y = mx + b and Linear Regression from high school. Here, we fitted a scatter plot to a line.
# Logistic Regression is another form of regression. 
# However, Logistic Regression helps us determine if a point should be in category A or B, which is a perfect fit.
l_r=LogisticRegression()
l_r.fit(X_train_trans,y_train)

# Validate that the model has learned something.
# Recall the model operates on vectors. First transform the test set using the vectorizer. 
# Then generate the predictions.
y_pred=l_r.predict(X_test_trans)

# We now want to see how we have done. We will be using three functions.
# `accuracy_score` tells us how well we have done. 
# 90% means that every 9 of 10 entries from the test dataset were predicted accurately.
# The `confusion_matrix` is a 2x2 matrix that gives us more insight.
# The top left shows us how many ham emails were predicted to be ham (that's good!).
# The bottom right shows us how many spam emails were predicted to be spam (that's good!).
# The other two quadrants tell us the misclassifications.
# Finally, the `classification_report` gives us detailed statistics which you may have seen in a statistics class.

print("Accuracy:",accuracy_score(y_test,y_pred))
print("CM: ",confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# Let's see which features (aka columns) the vectorizer created. 
# They should be all the words that were contained in the training dataset.
features=vectorizer.get_feature_names_out()

# You may be wondering what a machine learning model is tangibly. It is just a collection of numbers. 
# You can access these numbers known as "coefficients" from the coef_ property of the model
# We will be looking at coef_[0] which represents the importance of each feature.
# What does importance mean in this context?
# Some words are more important than others for the model.
# It's nothing personal, just that spam emails tend to contain some words more frequently.
# This indicates to the model that having that word would make a new email more likely to be spam.
coef=l_r.coef_[0]
features_df=pd.DataFrame({'Feature':features,'Coef':coef})

# Iterate over importance and find the top 10 positive features with the largest magnitude.
# Similarly, find the top 10 negative features with the largest magnitude.
# Positive features correspond to spam. Negative features correspond to ham.
# You will see that `http` is the strongest feature that corresponds to spam emails. 
# It makes sense. Spam emails often want you to click on a link.
sorted_coef=features_df.sort_values(by='Coef',ascending=False)
top_10_spam=sorted_coef.head(10)
top_10_ham=sorted_coef.tail(10)
print('Spam:\n',top_10_spam,'\n')
print('Ham:\n',top_10_ham)