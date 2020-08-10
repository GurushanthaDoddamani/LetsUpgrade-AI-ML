# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 10:25:40 2020

@author: dsgurushantha

"""

import pandas as pd

#import numpy as np

from sklearn import tree

from sklearn import preprocessing


general_data = pd.read_csv('general_data.csv')

# Business Travel conversion
label_encoder = preprocessing.LabelEncoder()

encoded_BusinessTravel = label_encoder.fit_transform(general_data["BusinessTravel"])

general_data["BusinessTravel"] = encoded_BusinessTravel

# Department Conversion

encoded_Department = label_encoder.fit_transform(general_data["Department"])

general_data["Department"] = encoded_Department

#Education Field Conversion
encoded_EducationField = label_encoder.fit_transform(general_data["EducationField"])

general_data["EducationField"] = encoded_EducationField

#Gender Conversion
encoded_Gender = label_encoder.fit_transform(general_data["Gender"])

general_data["Gender"] = encoded_Gender

#Job Role Conversion
encoded_JobRole = label_encoder.fit_transform(general_data["JobRole"])

general_data["JobRole"] = encoded_JobRole

#Mariatal Status Conversion
encoded_MaritalStatus = label_encoder.fit_transform(general_data["MaritalStatus"])

general_data["MaritalStatus"] = encoded_MaritalStatus

#Over18 Conversion
encoded_Over18 = label_encoder.fit_transform(general_data["Over18"])

general_data["Over18"] = encoded_Over18


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=1000,
                                  max_features=2,
                                  oob_score=True)
general_data.columns


features=["Age","BusinessTravel","DistanceFromHome","Education","EducationField","EmployeeCount",
          "Gender","JobLevel","JobRole","MaritalStatus","MonthlyIncome",
          "Over18","PercentSalaryHike","StandardHours","StockOptionLevel",
          "TrainingTimesLastYear","YearsAtCompany","YearsSinceLastPromotion","YearsWithCurrManager"]

rf_model.fit(X=general_data[features],y=general_data["Attrition"])

print("Oob Accrcy:")

print(rf_model.oob_score_)

for feature,imp in zip(features,rf_model.feature_importances_):
    print(feature,imp)
    
"""
Oob Accrcy:
1.0
Age 0.12047560439710457
BusinessTravel 0.03207041936760789
DistanceFromHome 0.0852381776239097
Education 0.047764944985995254
EducationField 0.05155304134835982
EmployeeCount 0.0
Gender 0.02006188598113334
JobLevel 0.044254030528405056
JobRole 0.06554262050663623
MaritalStatus 0.0449101643463438
MonthlyIncome 0.1114957321563401
Over18 0.0
PercentSalaryHike 0.07878221660990035
StandardHours 0.0
StockOptionLevel 0.04036385008498043
TrainingTimesLastYear 0.05288499328326655
YearsAtCompany 0.08579952309811327
YearsSinceLastPromotion 0.05038810526541759
YearsWithCurrManager 0.06841469041648601    
"""

"""
Important Features
Age
Monthly Income
"""

predictors = pd.DataFrame([general_data["Age"],general_data["MonthlyIncome"]]).T

tree_model = tree.DecisionTreeClassifier(max_depth=4)

tree_model.fit(X=predictors, y=general_data["Attrition"])

with open("DtreeAttrition.dot",'w') as f:
    f=tree.export_graphviz(tree_model,feature_names=["Age","MonthlyIncome"], out_file =f)

score = tree_model.score(X=predictors,y=general_data["Attrition"])  
  
#bank_test = pd.read_csv('testbank.csv')


test_preds = tree_model.predict(X= predictors)

Predicted_Output = pd.DataFrame({"EmployeeID":general_data["EmployeeID"],"Attrition":test_preds})

Predicted_Output.to_csv("Output_Attrition.csv",index=False)