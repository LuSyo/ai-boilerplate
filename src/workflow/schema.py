from pydantic import BaseModel, Field
from typing import List, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class PIIEntity(BaseModel):
  """Represents a piece of identified PII"""
  original_text: str
  start: int
  end: int
  label: str 
  redacted_as: str 

class PIIExtractionResult(BaseModel):
  """Container for all PII found in a single chunk"""
  entities: List[PIIEntity] = Field(default_factory=list)

class GraphState(BaseModel):
  seed: int = Field(default = 4)
  text: str = Field(default="")
  original_text: str = Field(default="")
  iterations: int = Field(default=0) # to limit nuber of iterations
  is_safe: bool = Field(default=False)
  pii_registry: List[PIIEntity] = Field(default_factory=list)
  messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)

  def __repr__(self):
        return f"GraphState(messages_count={len(self.messages)})"
  

