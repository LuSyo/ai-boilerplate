from typing import cast
from langgraph.graph import StateGraph, END, START
from workflow.schema import GraphState
from workflow.nodes import generate


def build_graph():
  workflow = StateGraph(GraphState)

  # Add nodes
  workflow.add_node("generate",generate)

  # Add edges
  workflow.add_edge(START, "generate")
  workflow.add_edge("generate", END)

  return workflow.compile()