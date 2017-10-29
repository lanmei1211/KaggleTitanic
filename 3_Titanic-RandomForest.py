#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:07:40 2017
Random Forest Third attempt
This scores  0.72248 which is not as good as the SVC attempt (0.76555)

@author: nb137
"""

import pandas as pd
from scipy.stats import mode
from sklearn import tree

dfall = pd.read_csv("train.csv")

# Drop name ticket cabin
# Use Class, sex, Age, SibSp, parch, fare, embarked 
df = dfall.drop(["Name", "Ticket", "Cabin"], axis=1)

# Deal with NaN and map 
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Sex"] = df["Sex"].map({'male':0, 'female':1})
df["Embarked"] = df["Embarked"].fillna(mode(df["Embarked"].dropna())[0][0])
df['Embarked'] = df['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int)
df = pd.concat([df, pd.get_dummies(df["Embarked"], prefix="Embarked")], axis=1)
df = df.drop(["Embarked"], axis=1)


model = tree.DecisionTreeClassifier()

X_train = df.drop(["Survived","PassengerId"], axis=1)
y_train = df["Survived"]

model.fit(X_train,y_train)

''' Now import the test data and predict with the model '''

dftest = pd.read_csv("test.csv")

# paste same process as above, will be overwriting df
df = dftest.drop(["Name", "Ticket", "Cabin"], axis=1)
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Sex"] = df["Sex"].map({'male':0, 'female':1})
df["Embarked"] = df["Embarked"].fillna(mode(df["Embarked"].dropna())[0][0])
df['Embarked'] = df['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int)
df = pd.concat([df, pd.get_dummies(df["Embarked"], prefix="Embarked")], axis=1)
df = df.drop(["Embarked"], axis=1)
# Oh they threw in one null fare. those bastards
df["Fare"] = df["Fare"].fillna(mode(df["Fare"].dropna())[0][0])

X_test = df.drop("PassengerId", axis=1)
y_test = model.predict(X_test)

output = pd.concat([df["PassengerId"],pd.DataFrame(y_test, columns=["Survived"])], axis=1)
output.to_csv("RandomForest-thirdTry.csv", index=False)