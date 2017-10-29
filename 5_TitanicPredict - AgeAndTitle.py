#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 28

This scored 0.78468, my highest score so far

Can use the following to choose a model
http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

Use Class, sex, Age, SibSp, parch, fare, embarked to predict
Add Title on top of this

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
df = df.drop(["Name", "Ticket", "Cabin"], axis=1)
df["Sex"] = df["Sex"].map({'male':0, 'female':1})

''' 
Train data has 177 nan ages, 2 nan embarked
ages 177 / 891 nan = 13.1%
So we only really have to worry about fixing NaN ages
'''

# So few embarked, lets just clear the NaNs out with the mode and then create the dummy variables
df["Embarked"] = df["Embarked"].fillna(mode(df["Embarked"].dropna())[0][0]) # Throws warning that nan values will be ignored, but seems OK
df['Embarked'] = df['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int)
df = pd.concat([df, pd.get_dummies(df["Embarked"], prefix="Embarked")], axis=1)
df = df.drop(["Embarked"], axis=1)
# Oh they threw in one null fare. those bastards
df["Fare"] = df["Fare"].fillna(mode(df["Fare"].dropna())[0][0])

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
                        "Mme":        "Regular",
                        "Mlle":       "Regular",
                        "Ms":         "Regular",
                        "Mr" :        "Regular",
                        "Mrs" :       "Regular",
                        "Miss" :      "Regular",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"
                        }       # Adapted from blog post to have fewer categories
df["Title"] = titles.map(TitleDict)
df = pd.concat([df.drop("Title", axis=1), pd.get_dummies(df["Title"], prefix="Title")], axis=1)

'''
Now thats we've cleaned data and predicted all missing ages let's take out
the test and train data.
Remembering that testing data is where Survived = NaN
'''
cleanTrain = df[df["Survived"].notnull()]
cleanTest = df[df["Survived"].isnull()].drop(["Survived"], axis=1)

# Train with SVC for now because it had an equally good fit to other stuff on my previous tests (0.785)

from sklearn.svm import SVC
model = SVC(kernel='linear')

X_train = cleanTrain.drop(["Survived","PassengerId"], axis=1)
y_train = cleanTrain["Survived"]

model.fit(X_train,y_train)

''' Now predict on test data '''

X_test = cleanTest.drop("PassengerId", axis=1)
y_test = model.predict(X_test)

output = pd.concat([cleanTest["PassengerId"],pd.DataFrame(y_test.astype(int), columns=["Survived"])], axis=1)
output.to_csv("5_AgeAndTitle.csv", index=False)
