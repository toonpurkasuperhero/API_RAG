import os, json
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

load_dotenv()

VSTORE_DIR = os.getenv("VSTORE_DIR", "data/vectorstore")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL_NAME = "mistral-large-2411"

embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    VSTORE_DIR,
    embed,
    allow_dangerous_deserialization=True
)

def retrieve_docs(query, k=3):
    results = vectorstore.similarity_search(query, k=k)
    return "\n".join([r.page_content for r in results])

def ask_mistral(prompt):
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    response = requests.post(MISTRAL_URL, json=data, headers=headers)
    if response.status_code == 200:
        resp = response.json()
        return resp['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code} - {response.text}"

def main():
    print("RAG Chat (Mistral LLM). Type 'exit' to quit.")
    while True:
        query = input("User: ")
        if query.lower() == "exit":
            break
        context = retrieve_docs(query)
        prompt = f"Use the following documentation to answer the question:\n{context}\n\nQuestion: {query}"
        answer = ask_mistral(prompt)
        print("AI:", answer)
        print("-"*50)

if __name__ == "__main__":
    main()
