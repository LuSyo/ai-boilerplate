from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from workflow.schema import GraphState, PersonaList, AuditReport
import pandas as pd

def generate_personas(state: GraphState, config: RunnableConfig):
  """
    Generate a batch of personas based on feedback from the user.
  """

  print("--- Generate personas ---")

  # Setup llm
  llm = config["metadata"].get("generator_llm")
  structured_llm = llm.with_structured_output(PersonaList)

  prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert demographic researcher. Generate 5 realistic synthetic citizen profiles."),
    ("human", "Current Guidance: {feedback}\n\nExisting count: {current_count}/{target}")
  ])
  
  chain = prompt | structured_llm
  new_batch = chain.invoke({
    "feedback": state.feedback,
    "current_count": len(state.personas),
    "target": state.target_count
  })

  return {
    "personas": new_batch.personas,
    "iterations": state.iterations + 1
  }

def audit_personas(state: GraphState, config: RunnableConfig):
  """Deterministic analysis followed by LLM-generated feedback."""
  print("--- Auditing Personas---")
  
  # 1. Deterministic Analysis (The "Hard Facts")
  # We use pandas to calculate the ground truth of our current dataset
      
  df = pd.DataFrame([p.model_dump() for p in state.personas])
  
  stats = {
    "current_count": len(df),
    "disability_rate": df["disability_status"].mean() * 100,
    "employment_rate": df["is_employed"].mean() * 100,
    "mean_income": df["annual_income"].mean(),
    "age_min": df["age"].min(),
    "age_max": df["age"].max()
  }

  llm = config["metadata"].get("audit_llm")
  structured_llm = llm.with_structured_output(AuditReport)

  prompt = ChatPromptTemplate.from_messages([
      ("system", """You are a Senior Demographic Auditor auditing a synthetic dataset. 
      Compare the dataset statistics against UK TARGETS:
      - Target Disability: 20%
      - Target Employment: 75%
      - Target Age Range: 18-80
      
      If a metric is off by more than 5%, mark is_balanced=False and 
      tell the Generator exactly what to focus on in the next batch."""),
      ("human", "Dataset Stats: {stats}\n\nProvide the Audit Report.")
  ])

  chain = prompt | structured_llm
  report = chain.invoke({"stats": stats})

  return {
      "feedback": report.feedback,
      "is_balanced": report.is_balanced
  }