import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from utils import Config

def rag_setup():
  load_dotenv()

  chunks = process_documents(Config.SOURCES_DIR)
  
  embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
  
  vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=f"{Config.DATA_DIR}/chroma_db"
  )
  

def process_documents(source_path: str):
  if not os.path.exists(source_path):
    print(f'Folder not found: {source_path}')
    return []
  
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
  )
  
  all_chunks = []
  for filename in os.listdir(source_path):
    try:
      with open(os.path.join(source_path, filename), 'r') as f:
        doc = f.read()
        chunks = text_splitter.create_documents([doc], metadatas=[{"source": filename}])
        all_chunks.extend(chunks)
    except Exception as e:
      print(f'Error processing file {filename}: {e}')
  
  return all_chunks

if __name__ == '__main__':
  rag_setup()