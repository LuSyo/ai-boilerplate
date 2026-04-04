from typing import cast
from langgraph.graph import StateGraph, END, START
from workflow.schema import GraphState
from workflow.nodes import generate_personas, audit_personas

def build_graph():
  workflow = StateGraph(GraphState)

  # Add nodes
  workflow.add_node("generate_personas", generate_personas)
  workflow.add_node("audit_personas", audit_personas)

  # Add edges
  workflow.add_edge(START, "generate_personas")
  workflow.add_edge("generate_personas", "audit_personas")
  workflow.add_conditional_edges("audit_personas", generate_more, {
    True: "generate_personas",
    False: END
  })

  return workflow.compile()

def generate_more(state: GraphState):
  if len(state.personas) < state.target_count and state.iterations < state.max_iterations:
    return True
  return False