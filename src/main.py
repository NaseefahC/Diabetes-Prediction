#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier

df = pd.read_csv("diabetes_prediction_dataset.csv")
df = df.drop_duplicates()
df = df[(df['age'] > 0) & (df['age'] <= 120)]
df = df[df['bmi'] <= 60]
df = df[(df['HbA1c_level'] >= 3) & (df['HbA1c_level'] <= 20)]
df = df[(df['blood_glucose_level'] >= 50) & (df['blood_glucose_level'] <= 400)]

df['smoking_history'] = df['smoking_history'].replace({'not current': 'former', 'ever': 'former'})
smoking_map = {'never': 0, 'former': 1, 'current': 2}
df['smoking_history_encoded'] = df['smoking_history'].map(smoking_map).fillna(4)

df = df[df['gender'].isin(['Female', 'Male'])]
gender_map = {'Female': 0, 'Male': 1}
df['gender_encoded'] = df['gender'].map(gender_map)

X = df[['gender_encoded', 'age', 'hypertension', 'heart_disease', 'smoking_history_encoded',
        'bmi', 'HbA1c_level', 'blood_glucose_level']]
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

xgb_model = XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    objective='binary:logistic',
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

y_probs = xgb_model.predict_proba(X_test)[:, 1]

thresholds = np.linspace(0.1, 0.5, 41)  
best_threshold = 0.5
best_f1 = 0

for t in thresholds:
    y_pred = (y_probs >= t).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    recall = report['1']['recall']
    f1 = report['1']['f1-score']
    
    if recall >= 0.80 and f1 > best_f1:
        best_threshold = t
        best_f1 = f1

y_pred_final = (y_probs >= best_threshold).astype(int)

print(f"Chosen threshold with recall >= 0.80: {best_threshold:.6f}")
print(classification_report(y_test, y_pred_final))
print(confusion_matrix(y_test, y_pred_final))
print("ROC-AUC Score:", roc_auc_score(y_test, y_probs))


# In[ ]:




