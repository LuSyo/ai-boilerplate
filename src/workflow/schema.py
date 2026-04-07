from pydantic import BaseModel, Field
from typing import List, Annotated, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ExplanationEvaluation(BaseModel):
  is_hallucination: bool = Field(default=False, description="Whether the explanation contains hallucinations.")
  score: int = Field(default=0, description="The safety score of the explanation, on a 0 to 5 scale.")
  feedback: str = Field(default="", description="Brief feedback about the explanation and its score.")

class GraphState(BaseModel):
  seed: int = Field(default = 4)
  messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
  patient_features: Dict[str, Any] = Field(default_factory=dict)
  raw_features: Dict[str, Any] = Field(default_factory=dict)
  risk_score: float = Field(default = 0.0)
  shap_values: Dict[str, Any] = Field(default_factory=dict)
  clinical_explanation: str = Field(default = "")
  evaluation: Optional[ExplanationEvaluation] = Field(default=None)

  def __repr__(self):
    return f"GraphState(risk_score={self.risk_score})"