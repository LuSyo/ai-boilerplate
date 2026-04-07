import os
import pickle
import pandas as pd
import shap
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from typing import cast
from pydantic import BaseModel, Field
from workflow.schema import GraphState, ExplanationEvaluation
from utils import Config
from train import group_diagnosis

def generate_risk_score(state: GraphState, config: RunnableConfig):
  """
    Loads the trained classifier and predicts the risk of readmission for given patient
  """
  print("--- Generate risk score ---")

  artifact_path = os.path.join(Config.RESULTS_DIR, "model_artifacts.pkl")
  with open(artifact_path, "rb") as f:
    artifacts = pickle.load(f)

  model = artifacts["model"]
  encoders = artifacts["encoders"]

  features = state.patient_features
  if not features:
    sample_df = artifacts["X_test"].iloc[0:1]
    raw_df = sample_df.copy()
    for col, encoder in encoders.items():
      if col in raw_df.columns:
        # Turn numbers back into categories (e.g., 4.0 -> "Circulatory")
        raw_df[col] = encoder.inverse_transform(raw_df[[col]]).flatten()
    raw_features_dict = raw_df.to_dict(orient='records')[0]
  else:
    raw_features_dict = features.copy()
    sample_df = pd.DataFrame([features])

    # group diagnosis
    for col in ['diag_1', 'diag_2', 'diag_3']:
      sample_df[col] = sample_df[col].apply(group_diagnosis)

    # Apply encoders
    categorical_cols = sample_df.select_dtypes(include=['object']).columns
    encoders = artifacts["encoders"]
    for col in categorical_cols:
      sample_df[col] = encoders[col].transform(sample_df[[col]])

  pred_proba = model.predict_proba(sample_df)

  return {
    "risk_score": float(pred_proba[0][1]),
    "patient_features": sample_df.to_dict(orient='records')[0], 
    "raw_features": raw_features_dict
  }

def get_shap_values(state: GraphState, config: RunnableConfig):
  """
    Conducts a SHAP analysis on the prediction for the given patient and returns the SHAP values
  """
  print("------ SHAP Analysis ------")

  artifact_path = os.path.join(Config.RESULTS_DIR, "model_artifacts.pkl")
  with open(artifact_path, "rb") as f:
    artifacts = pickle.load(f)

  model = artifacts["model"]
  background_data = artifacts["background_data"]

  sample_df = pd.DataFrame([state.patient_features])

  explainer = shap.TreeExplainer(model, data=background_data)
  shap_results = explainer.shap_values(sample_df)

  # SHAP output of shape (#num_samples, #num_features)
  # For each output, the SHAP values (summed across all features) plus the expected value equals the model’s output for that sample
  values = shap_results[0][:, 1] #sample 0, contribution to class 1, i.e. readmission

  shap_dict = dict(zip(sample_df.columns, [float(v) for v in values]))

  return { "shap_values": shap_dict}

def explain_risk(state: GraphState, config: RunnableConfig):
  """
    Formulates an explanation for the patient's risk prediction based on SHAP values and patient features.
  """

  print("--- Explain risk score ---")

  # sort by absolute impact
  sorted_values = sorted(
    state.shap_values.items(), 
    key=lambda x: abs(x[1]), 
    reverse=True
  )
  top_5 = sorted_values[:5]

  top_pos_list = [k for k, v in top_5 if v > 0]
  top_neg_list = [k for k, v in top_5 if v < 0]

  top_contributors = ", ".join(top_pos_list) if top_pos_list else "None identified"
  mitigating_factors = ", ".join(top_neg_list) if top_neg_list else "None identified"

  prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Senior Clinical Decision Support Assistant. 
    Your goal is to provide a 'Clinical Briefing' for a hospitalist to support 
    discharge planning and community referral. 

    CRITICAL RULES:
    1. Avoid technical jargon like 'SHAP' or 'random forest'. 
    2. Use professional medical terminology.
    3. Be concise (3-4 sentences total).
    4. Focus on 'Why' the risk is at its current level.
    """),
    ("human", """
    **Patient Risk Profile:**
    - Predicted 30-day Readmission Risk: {risk_score:.1%}
    - Primary Risk Drivers (Factors increasing risk): {top_contributors}
    - Mitigating Factors (Factors decreasing risk): {mitigating_factors}
    - Full Clinical Context: {features}

    **Task:**
    Provide a brief summary explaining the risk level and suggest if the patient 
    should be prioritized for community intervention upon discharge.
    """)
  ])

  llm = config["metadata"]["explain_risk_llm"]

  chain = prompt | llm | StrOutputParser()
  explanation = chain.invoke({
    "risk_score": state.risk_score,
    "top_contributors": top_contributors,
    "mitigating_factors": mitigating_factors,
    "features": state.raw_features
  })

  return {
    "clinical_explanation": explanation
  }

def evaluate_explanation(state: GraphState, config: RunnableConfig):
  """
  Acts as a 'Senior Clinical Auditor' to verify the generated explanation 
  against the actual SHAP values and raw features.
  """
  print("--- Evaluating for Hallucinations ---")
  
  # Use a high-quality model for judging (e.g., GPT-4o)
  llm = config["metadata"]["eval_llm"] 
  structured_llm = llm.with_structured_output(ExplanationEvaluation)

  prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Senior Clinical Auditor. Your task is to audit a 
    clinical briefing generated by an AI assistant.
    
    CRITICAL AUDIT RULES:
    1. Hallucinations: Is the briefing's risk assessment consistent with the numerical risk score? Does the briefing mention any medical conditions NOT found in the 'Actual Patient Data'?
    3. Accuracy: Does the briefing correctly identify the 'Top SHAP Drivers' as the main reasons for risk?
    """),
    ("human", """
    **Actual Patient Data:** {raw_features}
    **Numerical Risk Score:** {risk_score:.1%}
    **SHAP Drivers:** {shap_values}
    
    **AI Generated Briefing:**
    "{explanation}"
    
    **Your Task:**
    Evaluate the briefing. Flag the presence of hallucinations and return a score indicative of the clinical safety of the explanation, along with a brief feedback supporting your evaluation.
    """)
  ])

  # We use .with_structured_output if available, or simple parsing
  chain = prompt | structured_llm
  eval_raw = cast(ExplanationEvaluation, chain.invoke({
    "raw_features": state.raw_features,
    "risk_score": state.risk_score,
    "shap_values": state.shap_values,
    "explanation": state.clinical_explanation
  }))

  return { "evaluation": eval_raw}