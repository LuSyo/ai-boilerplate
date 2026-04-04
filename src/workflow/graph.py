from typing import cast
from langgraph.graph import StateGraph, END, START
from workflow.schema import GraphState
from workflow.nodes import detect_pii, redact_pii, audit_redaction


def build_graph():
  workflow = StateGraph(GraphState)

  # Add nodes
  workflow.add_node("detect_pii", detect_pii)
  workflow.add_node("redact_pii", redact_pii)
  workflow.add_node("audit_redaction", audit_redaction)

  # Add edges
  workflow.add_edge(START, "detect_pii")
  workflow.add_edge("detect_pii", "redact_pii")
  workflow.add_edge("redact_pii", "audit_redaction")
  workflow.add_conditional_edges("audit_redaction", should_continue, 
                                 {
                                   "rewrite": "redact_pii",
                                   "end": END
                                 })

  return workflow.compile()

def should_continue(state: GraphState):
  """
  Decides whether to finish or loop back for more redaction.
  """
  if state.is_safe or state.iterations >= 3:
    return "end"
  else:
    return "rewrite"