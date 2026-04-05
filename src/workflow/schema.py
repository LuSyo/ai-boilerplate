from pydantic import BaseModel, Field
from typing import List, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class GraphState(BaseModel):
  seed: int = Field(default = 4)
  category: str = Field(default="") # query, spam, urgent, unknown
  raw_message: str = Field(default="") #raw email text
  query: str = Field(default="") # query extracted from the email
  draft: str = Field(default="") # draft response
  subject: str = Field(default="") # subject of the draft email
  context: List[str] = Field(default_factory=list)
  requires_human_review: bool = Field(default=False)

  def __repr__(self):
    return f"GraphState(raw_message={self.raw_message[:50]})"