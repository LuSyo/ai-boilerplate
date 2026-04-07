from typing import cast
from langgraph.graph import StateGraph, END, START
from workflow.schema import GraphState
from workflow.nodes import explain_risk, generate_risk_score, get_shap_values, evaluate_explanation

def build_graph():
  workflow = StateGraph(GraphState)

  # Add nodes
  workflow.add_node("predict_risk", generate_risk_score)
  workflow.add_node("get_shap_values", get_shap_values)
  workflow.add_node("explain_risk", explain_risk)
  workflow.add_node("evaluate_explanation", evaluate_explanation)

  # Add edges
  workflow.add_edge(START, "predict_risk")
  workflow.add_edge("predict_risk", "get_shap_values")
  workflow.add_edge("predict_risk", "explain_risk")
  workflow.add_edge("explain_risk", "evaluate_explanation")
  workflow.add_edge("evaluate_explanation", END)

  return workflow.compile()