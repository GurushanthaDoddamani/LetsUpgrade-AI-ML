# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 09:07:42 2020

@author: dsgurushantha
"""


import pandas as pd

#import numpy as np

from sklearn import tree

#from sklearn import preprocessing

bank_data = pd.read_excel('Bank_Personal_Loan_Modelling.xlsx',sheet_name=1)

from sklearn.ensemble import RandomForestClassifier

bank_data.columns

rf_model = RandomForestClassifier(n_estimators=1000,
                                  max_features=2,
                                  oob_score=True)

features=["Age","Experience","Income","Family","CCAvg","Education","Mortgage","Securities Account",
          "CD Account","Online","CreditCard"]

rf_model.fit(X=bank_data[features],y=bank_data["Personal Loan"])

print("Oob Accrcy:")

print(rf_model.oob_score_)

for feature,imp in zip(features,rf_model.feature_importances_):
    print(feature,imp);

#Oob Accrcy:0.9874
#Age 0.045693250150168126
#Experience 0.044366730731085505
#Income 0.34367198972756674
#Family 0.09709929441776334
#CCAvg 0.1837484594475822
#Education 0.16030301098255967
#Mortgage 0.04508484148502315
#Securities Account 0.005793848684065506
#CD Account 0.055719202471053136
#Online 0.008538332347935544
#CreditCard 0.009981039555197203
    
#Important Features:Income, CCAvg, Education,
    
predictors = pd.DataFrame([bank_data["Income"],bank_data["CCAvg"],bank_data["Education"]]).T

tree_model = tree.DecisionTreeClassifier(max_depth=6)

tree_model.fit(X=predictors, y=bank_data["Personal Loan"])

with open("DtreeBank.dot",'w') as f:
    f=tree.export_graphviz(tree_model,feature_names=["Income","CCAvg","Education"], out_file =f)

score = tree_model.score(X=predictors,y=bank_data["Personal Loan"])  
  
#bank_test = pd.read_csv('testbank.csv')


test_preds = tree_model.predict(X= predictors)

Predicted_Output = pd.DataFrame({"Id":bank_data["ID"],"Loan Approved":test_preds})

Predicted_Output.to_csv("Output_bank.csv",index=False)

    