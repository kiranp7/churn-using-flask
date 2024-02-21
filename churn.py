from flask import Flask
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

df=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn (1).csv")
print(df.head())

df['class'] = df['Churn'].apply(lambda x : 1 if x == "Yes" else 0)

X = df[['tenure','MonthlyCharges']].copy()
y = df['class'].copy()

X_train, X_test, y_train, y_test = train_test_split( X,y , test_size = 0.2, random_state = 0)

clf = LogisticRegression(fit_intercept=True, max_iter=10000)
clf.fit(X_train, y_train)

pickle.dump(clf,open("model.pkl","wb"))