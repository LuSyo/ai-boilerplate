from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from workflow.schema import GraphState

def generate(state: GraphState, config: RunnableConfig):
  """
    Generate answer to a query based on context
  """

  print("--- Generate response ---")

  llm = config["metadata"]["generate_llm"]

  template = """
    Answer the query 

    Query: {query}

    Answer:
  """
  prompt = ChatPromptTemplate.from_template(template)

  last_query = state.messages[-1].content

  chain = prompt | llm | StrOutputParser()
  response = chain.invoke({"query": last_query})

  return {"messages": [("assistant", response)]}