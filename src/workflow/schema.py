from pydantic import BaseModel, Field
from typing import List, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class GraphState(BaseModel):
  seed: int = Field(default = 4)
  messages: Annotated[List[BaseMessage], add_messages] = Field(default_factory=list)
  context: List[str] = Field(default_factory=list)

  def __repr__(self):
    return f"GraphState(messages_count={len(self.messages)})"