# Explainable Clinical Triage System (Hybrid AI)

## **1\. Executive Summary**

The NHS is currently addressing a critical backlog in elective care. A significant portion of bed capacity is occupied by emergency readmissions that could have been prevented with targeted community intervention. While automated risk scoring exists, it is often dismissed by clinical staff as a "black box." Your task is to build a prototype system that not only predicts risk but provides a transparent, natural-language justification for every intervention suggested.

## **2\. Problem Statement**

Develop a hybrid pipeline that bridges structural data analysis with generative explanation. The goal is to move from a "Risk Score" to a "Clinical Briefing."

## **3\. Data**

https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

----

# my initial notes

Primary outcomes: reduce cost of emergency readmission by prioritising community intervention for identified patients. Support decisions for clinical staff to refer to community services upon discharge.

End user is a clinician. They need a risk score and trustworthy explanation to support their referral. 

Data sensitivity: no patient PII should be included in the output

Success metrics: precision (we want to avoid unnecessray referrals that would just move the workload to a different part of the system) and recall (we want to catch as many potential readmissions as possible). KPI: reduced cost in unplanned readmissions. 

Careful about data drift. Monitor and retrain regularly.

## Architecture:
### classifier
- train a classifier (e.g. RF) on a training set, cross validated on k-folds, to predict readmission under 30 days
- validate on holdout set
### explainer
graph:
input: a patient's features
- prediction node running the classifier (load the weights)
- SHAP node => results
- LLM explainer using the SHAP results and actual features to present the risk score and the supporting explaination to the clinician

----

# Classifier

## Run 1
Model trained. Accuracy: 0.59
Macro-averaged Recall: 0.42
Macro-averaged Precision: 0.54
F1 Score: 0.40
Classification Report:
              precision    recall  f1-score   support

           0       0.48      0.02      0.03      2285
           1       0.51      0.42      0.46      7117
           2       0.62      0.83      0.71     10952

    accuracy                           0.59     20354
   macro avg       0.54      0.42      0.40     20354
weighted avg       0.57      0.59      0.55     20354

=> balance the weights

## Run 2
Macro-averaged Precision: 0.54
F1 Score: 0.39
Classification Report:
              precision    recall  f1-score   support

           0       0.49      0.01      0.03      2285
           1       0.51      0.39      0.44      7117
           2       0.62      0.84      0.71     10952

    accuracy                           0.59     20354
   macro avg       0.54      0.41      0.39     20354
weighted avg       0.57      0.59      0.54     20354


Massive class imbalance
=> let's turn this into a binary classification problem for risk of readmission <30 days

## Run 3
Model trained. Accuracy: 0.89
Recall: 0.00
Precision: 0.50
F1 Score: 0.01

There must be a class imbalance in the training/test split

## Run 4
Training set size: 81412
Positive class in training set: 9086
Negative class in training set: 72326
Model trained. Accuracy: 0.89
Recall: 0.00
Precision: 0.65
F1 Score: 0.01

=> downsampling

## Run 5 
Training set size: 18172
Positive class in training set: 9086
Negative class in training set: 9086
Model trained. Accuracy: 0.63
Recall: 0.61
Precision: 0.17
F1 Score: 0.27

-----

# App

## 1/ Risk score node
load model, transfomr features if necessary, make prediction

## 2/ SHAP analysis node and explanation


see e2e/0

Improvement:
- sort shap values by absolute impact and use the top 5

=> see e2e/1

