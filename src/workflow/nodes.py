from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from workflow.schema import GraphState

class TriageOutput(BaseModel):
  category: str = Field(description="One of: 'query', 'urgent', 'spam', 'unknown'") 
  reasoning: str = Field(description="Brief explanation for the classification")
  is_sensitive: bool = Field(description="True if the email contains threats, crisis, or high-risk content")

def triage_node(state: GraphState, config: RunnableConfig):
  """
  Classifies the incoming email to determine the next step in the graph
  """
  email = state.raw_message
  
  # Bind the structured output to the model
  model = config['metadata']['triage_llm']
  structured_llm = model.with_structured_output(TriageOutput)
  
  # Perform the classification
  classification = structured_llm.invoke(f"Triage this citizen email: {email}")
  
  # Update the state
  return {
      "category": classification.category,
      "requires_human_review": classification.is_sensitive or classification.category == "urgent"
  }

def generate(state: GraphState, config: RunnableConfig):
  """
    Generate answer to a query based on context
  """

  print("--- Generate response ---")

  # Setup llm
  llm = config['metadata']['response_llm']

  template = """
    You are a civil servant in the St Albans City and District Council. Draft an email in response to the query:

    Query: {query}

    Draft response:
  """
  prompt = ChatPromptTemplate.from_template(template)

  query = state.raw_message #for now

  chain = prompt | llm | StrOutputParser()
  response = chain.invoke({"query": query})

  return {"draft": response}