# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 07:36:02 2020

@author: dsgurushantha
"""
import pandas as pd

import numpy as np

from sklearn import tree

from sklearn import preprocessing

titanic_train = pd.read_csv('train.csv')

age_var = np.where(titanic_train["Age"].isnull(),28,titanic_train["Age"])

titanic_train["Age"] = age_var

label_encoder = preprocessing.LabelEncoder()

encoded_sex = label_encoder.fit_transform(titanic_train["Sex"])

titanic_train["Sex"] = encoded_sex

predictors = pd.DataFrame([encoded_sex,titanic_train["Age"],titanic_train["Fare"]]).T

tree_model = tree.DecisionTreeClassifier(max_depth=6)

tree_model.fit(X=predictors, y=titanic_train["Survived"])

with open("Dtree.dot",'w') as f:
    f=tree.export_graphviz(tree_model,feature_names=["Sex","Age","Fare"], out_file =f)

score = tree_model.score(X=predictors,y=titanic_train["Survived"])  
  
titanic_test = pd.read_csv('test.csv')

new_age = np.where(titanic_test["Age"].isnull(),28,titanic_test["Age"])

titanic_test["Age"] = new_age

encoded_sex_test = label_encoder.fit_transform(titanic_test["Sex"])

test_features = pd.DataFrame([encoded_sex_test,titanic_test["Age"],titanic_test["Fare"]]).T

test_preds = tree_model.predict(X= test_features)

Predicted_Output = pd.DataFrame({"PassengerId":titanic_test["PassengerId"],"Survived":test_preds})

Predicted_Output.to_csv("Output.csv",index=False)


from sklearn.ensemble import RandomForestClassifier

titanic_train.columns

rf_model = RandomForestClassifier(n_estimators=1000,
                                  max_features=2,
                                  oob_score=True)

features=["Sex","Age","Fare"]

rf_model.fit(X=titanic_train[features],y=titanic_train["Survived"])

print("Oob Accrcy:")

print(rf_model.oob_score_)

for feature,imp in zip(features,rf_model.feature_importances_):
    print(feature,imp);

