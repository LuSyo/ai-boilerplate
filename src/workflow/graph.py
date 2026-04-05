from typing import cast
from langgraph.graph import StateGraph, END, START
from workflow.schema import GraphState
from workflow.nodes import generate, triage_node


def build_graph():
  workflow = StateGraph(GraphState)

  # Add nodes
  workflow.add_node("triage", triage_node)
  workflow.add_node("generate", generate)

  # Add edges
  workflow.add_edge(START, "triage")
  workflow.add_conditional_edges("triage", triage_routing, {
    "spam": END,
    "review": END,
    "retrieve": "generate" #for now, just one node to generate a response; later: query identification node, retrieval node, then response if relevant chunk retrieved
  })
  workflow.add_edge("generate", END)

  return workflow.compile()

def triage_routing(state: GraphState):
  if state.category == "spam":
    return "spam"
  elif state.category == "urgent":
    return "review"
  else:
    return "retrieve"