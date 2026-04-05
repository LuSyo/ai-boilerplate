from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from workflow.schema import GraphState, TriageOutput


def triage_node(state: GraphState, config: RunnableConfig):
  """
  Classifies the incoming email to determine the next step in the graph
  """
  print("--- Triage email ---")

  email = state.raw_message
  
  # Bind the structured output to the model
  model = config['metadata']['triage_llm']
  structured_llm = model.with_structured_output(TriageOutput)
  
  # Perform the classification
  classification = structured_llm.invoke(f"Triage this citizen email: {email}")
  
  print("Classification: ", classification.category)
  print("Reasoning: ", classification.reasoning)
  print("Sensitive: ", classification.is_sensitive)

  # Update the state
  return {
      "category": classification.category,
      "requires_human_review": classification.is_sensitive or classification.category == "urgent"
  }

def query_extractor(state: GraphState, config: RunnableConfig):
  """
    Extract the query from the email
  """

  # TO IMPLEMENT

  return state

def retrieve_documents(state: GraphState, config: RunnableConfig):
  """
    Retrieve most relevant documents for the query
  """
  print("--- Retrieve documents ---")

  query = state.raw_message

  embeddings = config['metadata']['embeddings']
  vector_store = Chroma(persist_directory="./data/chroma_db", embedding_function=embeddings)

  retriever = vector_store.as_retriever(search_kwargs={"k": 3})
  docs = retriever.invoke(query)

  context = [doc.page_content for doc in docs]

  return {"context": context}

def generate(state: GraphState, config: RunnableConfig):
  """
    Generate answer to a query based on context
  """

  print("--- Generate response ---")

  # Setup llm
  llm = config['metadata']['response_llm']

  template = """
    You are a civil servant in the St Albans City and District Council. Draft an email in response to the query, based on the provided context:

    Query: {query}

    Context: {context}

    Draft response:
  """
  prompt = ChatPromptTemplate.from_template(template)

  query = state.raw_message #for now

  chain = prompt | llm | StrOutputParser()
  response = chain.invoke({"query": query, "context": state.context})

  return {"draft": response}