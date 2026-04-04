# Anonymisation

The Problem: You are given a set of 50 semi-structured documents (e.g., FOI requests or medical notes). You must build a pipeline that identifies and redacts Personally Identifiable Information (PII) while preserving the context for a downstream summary.

Initial thoughts:

We will need the following nodes:

- process documents: PDFs => overlapping chunks (because they might be too long?)
- PII detector script (I'm assuming there are some solutions already for simple PII like phone numbers, addrsses, etc...)
- LLM node to detect remaining PII that might be less obvious
- Replace the identified strings in the chunks: I'm wondering what the strategy should be here, maybe just inserting "###"
- LLM check for lost meaning and remaining PII? => Self-correcting loop
- Pass the chunks to the rest of the pipeline

Better strategy for replacing PII without losing meaning: [PATIENT_ADDRESS]

---

## First run

- record IDs not anonymised (dispatch record in doc 1)
- phone number missed in doc 1
- missed doctor name in doc 1
- missed clinic location in doc 1
- meaning seems to be preserved

=> implement self correction loop

PROMPT:
"You are a Senior Privacy Auditor. Inspect the following REDACTED text for any LEAKED PII.
LEAKS are names, dates, locations, contact details, personal attributes, treatment staff details, record IDs, etc. that are NOT inside brackets like [PATIENT_1] that could be used to re-identify the patient.
If you find a leak, return it. If the text is perfectly safe, return an empty list."

## Second run

Better but still missed the record ID, the phone number and doctor house
in document 2, also misses unique traits and events that could re-identify the patient

=> improve the prompt to look for specific targets

NEW PROMPT:
  "You are a Senior Privacy Auditor. Inspect the following REDACTED text for any LEAKED PII that could be used to re-identify the patient.

  CRITICAL TARGETS:
  - Alphanumeric IDs for clinical records
  - Professional staff names
  - Phone numbers in any format
  - Rare medical traits or events that would identify the patient

  RULES:
  - For any leak found, suggest a label like [LABEL_NUMBER] (e.g. [DOCTOR_2],[PATIENT_1_EVENT], [DOCTOR_1_NAME])
  - Ensure labels match existing numbering if possible.
  
  If the text is perfectly safe, return an empty list."

## Third run

Still missing some tricky PII, e.g.:
"He noted that the stress of the ongoing 2026 dockworkers' strike, of which he is the [PATIENT_1_JOB_TITLE_2] for the management side, has exacerbated his symptom"
= still identifiable

maybe adding to the prompt "could be reidentified with a quick search based on recent news?" or something like that

instead of processing the redacted text, the auditor should maybe process the PII registry on the raw text and check that they are appropriate and not missing tricky PII => they can replace ones that arent' covering enough information... Might be tricky

Still didn't catch the phone number. Would need few shots prompt

---

# Eval

Need to test recall and precision

BLEU and ROUGE won't work as the LLM might label the redactions differently

We check deterministically if the app has replaced text by labels in the right spots.

Problem with first implementation: we check the coordinates agaisnt the ground truth, where labels are present, but as the labels in the response might be longer or shorter, there's a shift in coordinates => we should use the coordinates in the original text, or maybe have a ground truth list of coordinates, and check them against the coordinates stored by the model in the PII registry?
Otherwise, lower recall and precision 
