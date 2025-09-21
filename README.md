# **API\_RAG: Retrieval-Augmented Generation Chatbot**

API\_RAG is a **Retrieval-Augmented Generation (RAG)** system that lets you query API documentation or any web pages you’ve ingested.
It combines:

* **LangChain + FAISS** for document ingestion and vector search
* **Sentence-Transformer embeddings** for semantic retrieval
* **Mistral LLM** for natural-language answers
* Optional **Streamlit web UI** for a friendly chat interface

---

## Features

* **Automated Web Ingestion**
  Scrapes web pages (API docs, HTML, etc.) and splits them into manageable chunks.

* **Vector Database**
  Uses FAISS to store and quickly retrieve semantically similar text.

* **RAG Pipeline**
  Combines the retrieved context with the user query to provide grounded answers.

* **LLM Back-End**
  Uses Mistral’s `chat/completions` endpoint for generation.

* **Flexible Interfaces**

  * **CLI Chat**: Simple terminal interface (`python src/chat.py`)
  * **Streamlit UI**: Web-based interface (`streamlit run src/streamlit_app.py`)

---

## Project Structure

```
API_RAG/
├─ src/
│  ├─ ingestion/
│  │   └─ build_vectorstore.py   # Crawl & build FAISS index
│  ├─ chat.py                    # CLI chat interface
│  └─ streamlit_app.py           # Streamlit web app
├─ data/
│  └─ vectorstore/               # Saved FAISS index (generated)
├─ .env                          # Environment variables
├─ requirements.txt
└─ README.md                      # (this file)
```

---

## Installation

1. **Clone the repo**

```bash
git clone https://github.com//API_RAG.git
cd API_RAG
```

2. **Create & activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

Key packages:

* `langchain-community`
* `faiss-cpu`
* `sentence-transformers`
* `streamlit`
* `requests`
* `beautifulsoup4`
* `python-dotenv`

---

## Environment Variables

Create a `.env` file in the project root:

```env
VSTORE_DIR=data/vectorstore
MISTRAL_API_KEY=your_mistral_api_key
```

*Optional:*
`USER_AGENT` to identify web requests when scraping.

---

## Building the Vector Store

Before chatting, ingest documentation:

```bash
python src/ingestion/build_vectorstore.py
```

This will:

1. Load URLs or HTML files you specify inside `build_vectorstore.py`.
2. Fetch & clean the pages using BeautifulSoup.
3. Split text into chunks with `RecursiveCharacterTextSplitter`.
4. Embed each chunk with `sentence-transformers/all-MiniLM-L6-v2`.
5. Store vectors in FAISS (`data/vectorstore`).

> ⚠**Note**: Edit `urls` in `build_vectorstore.py` to point to your docs.

---

## Command-Line Chat

Once the vectorstore is built:

```bash
python src/chat.py
```

* Type your question and press Enter.
* Type `exit` to quit.

Example:

```
User: How do I fetch API data?
AI: You can use the GET /endpoint...
```

---

## Streamlit Web UI

For a richer experience:

```bash
streamlit run src/streamlit_app.py
```

* Opens a browser interface (default: [http://localhost:8501](http://localhost:8501)).
* Enter questions in a text box and view the LLM’s response.

---

## How It Works

1. **Query**
   User asks a question.

2. **Retriever**
   FAISS finds the top-k text chunks semantically similar to the query.

3. **Prompt Construction**
   Retrieved text + user question → prompt.

4. **Mistral API**
   Sends prompt to Mistral `chat/completions` endpoint.

5. **Response**
   Streamlit/CLI displays the generated answer.

---

## Customization

* **Change Embeddings**
  Update the `model_name` in both `build_vectorstore.py` and `chat.py`.

* **Switch LLM**
  Replace the `ask_mistral` function in `chat.py` or `streamlit_app.py` with another provider (OpenAI, Groq, etc.).

* **Add New Data**
  Re-run `build_vectorstore.py` after editing the URLs or adding local files.

---

## Testing

* Unit test retrieval by running a known query and checking if expected context is present.
* Validate that Mistral API key works with a simple POST request.

---

## Troubleshooting

* **Missing packages**:
  `pip install beautifulsoup4 sentence-transformers`

* **Deprecation Warnings**:
  LangChain recently migrated imports (e.g., `langchain_community`). Use the updated paths already included.

* **Pickle Deserialization**:
  FAISS loading requires `allow_dangerous_deserialization=True` if you trust your own index.

* **Invalid Model**:
  Ensure `MODEL_NAME` matches an available Mistral model (e.g., `mistral-large-2411`).

---

## License

MIT License – Feel free to fork, modify, and distribute.

---

## Contributing

1. Fork the project
2. Create a feature branch: `git checkout -b feature/awesome`
3. Commit changes: `git commit -m "Add awesome feature"`
4. Push to branch: `git push origin feature/awesome`
5. Open a Pull Request
