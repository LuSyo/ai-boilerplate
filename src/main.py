from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from utils import parse_args, set_global_seeds, setup_logger, Config
from workflow.graph import build_graph
from workflow.schema import GraphState
from markitdown import MarkItDown
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from collections import defaultdict


def main():
  load_dotenv()
  args = parse_args()
  set_global_seeds(args.seed)
  logger = setup_logger(Config.LOG_DIR, args.exp_name)

  app = build_graph()

  if args.query:
    texts_to_process = [args.query]
    logger.info(f"Running manual redaction on input text...")
  else:
    # BATCH RUN: Process all documents in the folder
    docs = process_documents(Config.SOURCES_DIR, logger)
    texts_to_process = [doc.page_content for doc in docs]

  for i, text in enumerate(texts_to_process):
    logger.info(f"--- PROCESSING CHUNK {i} ---")

    initial_state = GraphState(
      messages=[],
      text=text,
      original_text=text,
      pii_registry=[],
      seed=args.seed
    )

    final_state = app.invoke(initial_state)

    docs[i].page_content = final_state["text"]

  save_results(docs, args.exp_name)

def process_documents(folder_path: str, logger):
  if not os.path.exists(folder_path):
    print(f'Folder not found: {folder_path}')
    return []
  
  md = MarkItDown()

  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
  )

  all_chunks = []

  for filename in os.listdir(folder_path):
    if filename.startswith('.'): continue
    full_path = os.path.join(folder_path, filename)

    try:
      result = md.convert(full_path)
      markdown_text = result.text_content
      chunks = text_splitter.create_documents(
        texts=[markdown_text],
        metadatas=[{"source": filename}]
      )
      
      all_chunks.extend(chunks)
    except Exception as e:
      logger.error(f"Error processing {filename}: {e}")

  return all_chunks


def save_results(chunks: list, exp_name: str):
  result_path = os.path.join(Config.RESULTS_DIR, exp_name)
  os.makedirs(result_path, exist_ok=True)

  redacted_docs = defaultdict(list)
  
  for chunk in chunks:
    # Get the filename from metadata, default to 'manual_run.txt'
    source = chunk.metadata.get("source", "manual_run.txt")
    redacted_docs[source].append(chunk.page_content)

  for filename, content_parts in redacted_docs.items():
    full_text = "\n\n".join(content_parts)
    save_path = os.path.join(result_path, f"redacted_{filename}.txt")
    
    with open(save_path, "w", encoding="utf-8") as f:
      f.write(full_text)



if __name__ == "__main__":
  main()