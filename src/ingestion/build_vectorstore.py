import os, glob, json
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

load_dotenv()

DATA_DIR = "data/raw"
VSTORE_DIR = os.getenv("VSTORE_DIR", "data/vectorstore")
DOC_URL = os.getenv("DOC_URL", "https://api.freshservice.com/#ticket_attributes")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VSTORE_DIR, exist_ok=True)

def scrape_docs():
    loader = WebBaseLoader(DOC_URL)
    docs = loader.load()
    for i, doc in enumerate(docs):
        with open(os.path.join(DATA_DIR, f"doc_{i}.json"), "w", encoding="utf-8") as f:
            json.dump({"content": doc.page_content, "source": doc.metadata.get("source")}, f)
    return docs

def load_docs():
    docs = []
    for path in glob.glob(os.path.join(DATA_DIR, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        docs.append(Document(page_content=j.get("content",""), metadata={"source": j.get("source")}))
    return docs

def main():
    if not os.listdir(DATA_DIR):
        scrape_docs()
    
    docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    texts = splitter.split_documents(docs)

    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embed)

    vectorstore.save_local(VSTORE_DIR)
    print(f"Saved vectorstore to {VSTORE_DIR}, {len(texts)} chunks.")

if __name__ == "__main__":
    main()