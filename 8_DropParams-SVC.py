#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 28

This scored 0.77033, lower than my attempt using SVC with Title

Follow previous attempt 5 strategies, but drop some parameters in hopes of better fit

Use Sex, title(parsed and classified), age, fare.
Keep SibSp, parch, class as they also have input, but we might be able to do without them later.
Drop embarked, ticket.

Use Class, sex, Age, SibSp, parch, fare, embarked to predict
Add Title on top of this

Then use sklearn to select our parameters

And use Random Forest with parameters found in a gridsearch in the blog post

Title parsing influenced from these two sources
http://ahmedbesbes.com/how-to-score-08134-in-titanic-kaggle-challenge.html    last accessed 10/28/17
https://github.com/ramansah/kaggle-titanic/blob/master/Analysis.ipynb    last accessed 10/28/17

Use SVC to predict the final output

Use SVR to predict unknown ages


@author: nb137
"""

import pandas as pd
from scipy.stats import mode

dftrain = pd.read_csv("train.csv")
dftest = pd.read_csv("test.csv")

# Put them together in order to do data munging at the same time
df = pd.concat([dftrain, dftest], axis=0)

# Grab name to use later
names = df["Name"]
#Drop categories we wont use for fit
df = df.drop(["Name", "Ticket", "Cabin", "Embarked"], axis=1)
df["Sex"] = df["Sex"].map({'male':0, 'female':1})

''' 
Train data has 177 nan ages, 2 nan embarked
ages 177 / 891 nan = 13.1%
So we only really have to worry about fixing NaN ages
'''

# Oh they threw in one null fare. those bastards
df["Fare"] = df["Fare"].fillna(mode(df["Fare"].dropna())[0][0])


'''
Pull out titles and set them to categories
Will create name categories and then set dummy variables for each title
'''
titles = names.map(lambda name: name.split(',')[1].split('.')[0].strip())
TitleDict = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"
                        }       # Adapted from blog post to have fewer categories
df["Title"] = titles.map(TitleDict)
df = pd.concat([df.drop("Title", axis=1), pd.get_dummies(df["Title"], prefix="Title")], axis=1)

# We can't use survived as a predictor because the test data doesnt have it
# If we only used the train data with survive we wouldnt be able to apply it to the test data, because it wouldn't have the survive degree of fredom
XageTrain = df.dropna(subset=["Age"]).drop(["Age", "PassengerId", "Survived"], axis=1)
yageTrain = df.dropna(subset=["Age"])["Age"]

Xage = df[df["Age"].isnull()].drop(["Age", "PassengerId", "Survived"], axis=1)
from sklearn.linear_model import Lasso
ageModel = Lasso()
ageModel.fit(XageTrain, yageTrain)
yage = ageModel.predict(Xage)
# This returns some negative ages. Let's turn any negative age into 1 to show very young
yage[yage < 0] = 1
# Set all Age values that are null to the yage values
df.loc[df["Age"].isnull(), "Age"] = yage

'''
Now thats we've cleaned data and predicted all missing ages let's take out
the test and train data.
Remembering that testing data is where Survived = NaN
'''
cleanTrain = df[df["Survived"].notnull()]
cleanTest = df[df["Survived"].isnull()].drop(["Survived"], axis=1)

from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel

# Model run to select most important parameters

X_train = cleanTrain.drop(["Survived","PassengerId"], axis=1)
y_train = cleanTrain["Survived"]
X_test = cleanTest.drop("PassengerId", axis=1)


selectionModel = SVC(C=10, kernel='linear')

# Use the parameter model fit to narrow down to important params
# Do for both fitting and testing parameters
selectionModel.fit(X_train,y_train)
model = SelectFromModel(selectionModel, prefit=True)
X_train_reduced = model.transform(X_train)
X_test_reduced = model.transform(X_test)

# Use best params from blog post, maybe implement grid search later
model = SVC(C=10,kernel='linear')
model.fit(X_train_reduced,y_train)
reducedScore = model.score(X_train_reduced,y_train) # = 0.8485
# model score for non-reduced was 0.8619
y_test = model.predict(X_test_reduced)

output = pd.concat([cleanTest["PassengerId"],pd.DataFrame(y_test.astype(int), columns=["Survived"])], axis=1)
output.to_csv("8_dropParamsSVC.csv", index=False)
