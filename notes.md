# Automated "Triage" of Citizen Correspondence
**The Problem:** Thousands of emails arrive daily. Some are urgent (threats, crisis), some are policy queries, and some are spam.
**The Task:** Build a multi-stage classifier. Stage 1: Route the email. Stage 2: Extract key entities (Location, Department). Stage 3: Draft a suggested response based on a "Knowledge Base" file provided.
**The Assessment:** How do you handle "adversarial" or "confused" emails? If the model is unsure, how does your code handle the "Unknown" category?

# Initial thoughts

not sure how to route the emails
MVP: take email formats (eml) as input
RAG with policy documents; ingestion pipeline with chunk pslitter and ChromaDB
Triage goals:
- only address emails with queries (even if angry); 
- if unsure or seems urgent, flag for human review; What's sensitive? (in real scenario, would map this out with departments)
- provide accurate, relevant information
- don't invent or twist information (avoid hallucinations)
- if no info availalbe, flag for human review. 
- be polite and helpful

Triage workflow:
- Router node to only address emails with queries, filter out spam, flag urgent emails for human review (maybe two nodes, specialised for urgent emails and spam)
- Answer query node:
  - Formulate the query from the email: what does the user want to know? what information could help them understand their situation?
  - Retrieve sources most relevant to the query
  - Formulate a response answerign the query and addressing the user email


