from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import cast
from workflow.schema import GraphState, PIIEntity, PIIExtractionResult

def detect_pii(state: GraphState):
  """Scans raw text for PII entities"""
  print(f"---DETECT---")
  llm = ChatOpenAI(model="gpt-4o", temperature=0)
  structured_llm = llm.with_structured_output(PIIExtractionResult)

  system_prompt = """
  You are a high-security data privacy officer. 
  Identify all PII in the provided medical text.
  Provide the exact start and end character offsets.
  Propose redactions for each PII entity as [LABEL_NUMBER] (e.g., [PATIENT_1], [DOCTOR_1], [PATIENT_1_ADDRESS_1]) to preserve utility of the medical text.
  """

  new_registry = list(state.pii_registry)

  result = cast(PIIExtractionResult, structured_llm.invoke([
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": f"TEXT: {state.text}"}
  ]))
    
  new_registry.extend(result.entities)

  return {"pii_registry": new_registry}

def redact_pii(state: GraphState):
  """Applies the PII registry to the text to created redacted text"""
  print("---REDACT---")
  redacted_text = state.original_text

  for entity in state.pii_registry:
    redacted_text = redacted_text.replace(
      entity.original_text, entity.redacted_as
    ) # look for the original text and replace it wit hthe redaction

  return {"text": redacted_text}

def audit_redaction(state: GraphState):
  """Inspects redacted text for leaked PII"""
  print("---AUDIT REDACTION---")
  llm = ChatOpenAI(model="gpt-4o", temperature=0)
  structured_llm = llm.with_structured_output(PIIExtractionResult)

  system_prompt = """
  You are a Senior Privacy Auditor. Inspect the following REDACTED text for any LEAKED PII that could be used to re-identify the patient.

  CRITICAL TARGETS:
  - Alphanumeric IDs for clinical records
  - Professional staff names
  - Phone numbers in any format
  - Rare medical traits or events that would identify the patient

  RULES:
  - For any leak found, suggest a label like [LABEL_NUMBER] (e.g. [DOCTOR_2],[PATIENT_1_EVENT], [DOCTOR_1_NAME])
  - Ensure labels match existing numbering if possible.
  
  If the text is perfectly safe, return an empty list.
  """

  result = cast(PIIExtractionResult, structured_llm.invoke([
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"REDACTED TEXT: {state.text}"}
  ]))

  is_safe = len(result.entities) == 0

  new_registry = list(state.pii_registry)
  new_registry.extend(result.entities)

  return {
    "pii_registry": new_registry, 
    "is_safe": is_safe, 
    "iterations": state.iterations + 1
  }