from pydantic import BaseModel, Field
from typing import List, Annotated
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class Persona(BaseModel):
  annual_income: float = Field(default = 0.0, description="Gross annual income in GBP")
  household_size: int = Field(default = 0, description="Number of people living in the household")
  disability_status: bool = Field(default = False, description="Whether the individual has a registered disability")
  is_employed: bool = Field(default = False, description="Current employment status")
  age: int = Field(default = 0, description="Age of the individual")

class PersonaList(BaseModel):
  """A container for a batch of generated personas."""
  personas: List[Persona]

class AuditReport(BaseModel):
  """Report from the auditor regarding demographic balance."""
  feedback: str = Field(description="Specific instructions for the generator to improve balance.")
  is_balanced: bool = Field(description="Whether the current dataset meets demographic targets.")


class GraphState(BaseModel):
  seed: int = Field(default = 4)
  personas: Annotated[List[Persona], operator.add] = Field(default_factory=list)
  target_count: int = Field(default = 100)
  feedback: str = Field(default = "")
  iterations: int = Field(default = 0)
  max_iterations: int = Field(default = 20)
  is_balanced: bool = Field(default = False)

  def __repr__(self):
    return f"GraphState(personas_count={len(self.personas)})"