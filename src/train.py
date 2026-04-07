import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.utils import resample
from utils import Config


def train_classifier():
  # Download the dataset
  dataset = pd.read_csv(Config.DATA_DIR + '/cleaned_dataset.csv')

  # categorical columns: Ordinal Encoding
  for col in ['diag_1', 'diag_2', 'diag_3']:
    dataset[col] = dataset[col].apply(group_diagnosis)

  categorical_cols = dataset.select_dtypes(include=['object']).columns
  encoders_dict = {}
  for col in categorical_cols:
    oe = OrdinalEncoder()
    dataset[col] = oe.fit_transform(dataset[[col]])
    encoders_dict[col] = oe

  X = dataset.drop("readmitted", axis=1)
  y = dataset["readmitted"]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

  train_data = pd.concat([X_train, y_train], axis=1)
  negative_class = train_data[train_data.readmitted == 0]
  positive_class = train_data[train_data.readmitted == 1]
  neg_downsampled = resample(negative_class, 
                               replace=False,   
                               n_samples=len(positive_class),
                               random_state=Config.SEED)
  
  balanced_train = pd.concat([neg_downsampled, positive_class])
  X_train_balanced = balanced_train.drop('readmitted', axis=1)
  y_train_balanced = balanced_train['readmitted']

  print("Training set size:", y_train_balanced.shape[0])
  print("Positive class in training set:", y_train_balanced.sum())
  print("Negative class in training set:", y_train_balanced.shape[0] - y_train_balanced.sum())

  # Train the classifier
  clf = RandomForestClassifier(n_estimators=100, random_state=Config.SEED)
  clf.fit(X_train_balanced, y_train_balanced)

  artifacts = {
    "model": clf,
    "encoders": encoders_dict,
    "feature_names": list(X.columns),
    "X_test": X_test,
    "y_test": y_test,
    "background_data": X_train.sample(100)
  }

  os.makedirs(Config.RESULTS_DIR, exist_ok=True)
  with open(os.path.join(Config.RESULTS_DIR, "model_artifacts.pkl"), "wb") as f:
      pickle.dump(artifacts, f)

  y_pred = clf.predict(X_test)

  print(f"Model trained. Accuracy: {clf.score(X_test, y_test):.2f}")
  print(f"Recall: {recall_score(y_test, y_pred):.2f}")
  print(f"Precision: {precision_score(y_test, y_pred):.2f}")
  print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")

def group_diagnosis(value):
  """Maps ICD-9/10 codes to broad clinical categories."""
  if pd.isna(value) or value == '?' or value == 'Unknown':
      return 'No Diagnosis'
  
  try:
      if str(value).startswith(('V', 'E')):
          return 'Other'
      
      code = float(value)
      
      if 1 <= code < 140:
          return 'Infection'
      elif 140 <= code < 240:
          return 'Neoplasm'
      elif 240 <= code < 250 or 251 <= code < 280:
          return 'Endocrine'
      elif 250 <= code < 251: # Exact match for Diabetes
          return 'Diabetes'
      elif 280 <= code < 290:
          return 'Blood'
      elif 290 <= code < 320:
          return 'Mental'
      elif 320 <= code < 390:
          return 'Nervous'
      elif 390 <= code < 460 or code == 785:
          return 'Circulatory'
      elif 460 <= code < 520 or code == 786:
          return 'Respiratory'
      elif 520 <= code < 580 or code == 787:
          return 'Digestive'
      elif 580 <= code < 630 or code == 788:
          return 'Genitourinary'
      elif 630 <= code < 680:
          return 'Obstetric'
      elif 680 <= code < 710:
          return 'Skin'
      elif 710 <= code < 740:
          return 'Musculoskeletal'
      elif 740 <= code < 760:
          return 'Congenital'
      elif 760 <= code < 780:
          return 'Perinatal'
      elif 800 <= code <= 999:
          return 'Injury'
      else:
          return 'Other'
  except ValueError:
    return 'Other'


if __name__ == "__main__":
  train_classifier()


