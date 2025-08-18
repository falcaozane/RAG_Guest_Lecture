# Langchain PDF Chatbot 🤖💬

## Overview

This project is a chatbot that answers questions based on a provided PDF document. It uses the Langchain framework to perform a Retrieval-Augmented Generation (RAG) task. The chatbot processes the content of a PDF file, creates a vector store for efficient searching, and uses a large language model (LLM) from Groq to generate answers to user queries.

***

## 🏗️ Architecture

The chatbot's architecture can be summarized in the following steps:

1.  **Document Loading:** The project uses `PyPDFLoader` from `langchain-community` to load a PDF file named `Blockchain_Guest_Lecture_Notes.pdf`.
2.  **Text Splitting:** The loaded document is divided into smaller, manageable chunks using `RecursiveCharacterTextSplitter`.
3.  **Embedding Creation:** `HuggingFaceEmbeddings` with the `all-MiniLM-L6-v2` model is used to convert the text chunks into numerical vectors.
4.  **Vector Store:** The embeddings are stored in a `Chroma` vector store for efficient semantic search.
5.  **Retrieval:** A retriever is created from the vector store to find the most relevant document chunks based on a user's query.
6.  **Language Model (LLM):** The `ChatGroq` model, specifically `llama-3.1-8b-instant`, is used to generate the final answer. The model is accessed using an API key loaded from a `.env` file.
7.  **RAG Chain:** A retrieval chain is created to combine the retrieved context with the user's question and pass it to the LLM to generate a concise answer.

The entire process is illustrated in the table below:

| Component | Function |
| :--- | :--- |
| `PyPDFLoader` | Loads the PDF document. |
| `RecursiveCharacterTextSplitter` | Breaks down the document into chunks. |
| `HuggingFaceEmbeddings` | Creates numerical representations (embeddings) of the text. |
| `Chroma` | Stores the embeddings in a searchable vector database. |
| `ChatGroq` | The large language model that generates the final response. |
| `create_retrieval_chain` | Connects all the components to form the RAG pipeline. |

***

## ⚙️ Installation

To set up the project, you need to install the required dependencies:

```bash
pip install -r requirements.txt

---
### 📝 Requirements

The project dependencies are listed in `requirements.txt`:

* `langchain-community`
* `pypdf`
* `langchain-huggingface`
* `langchain-chroma`
* `sentence-transformers`
* `langchain-groq`
* `python-dotenv`

---
### 🚀 Usage

1.  **Set up your API Key:** Create a `.env` file in the root directory of your project and add your Groq API key.

    ```ini
    GROQ_API_KEY="YOUR_API_KEY_HERE"
    ```

2.  **Place your PDF:** Ensure you have a PDF file named `Blockchain_Guest_Lecture_Notes.pdf` inside a `data/` directory.

3.  **Run the application:** Execute the main script from your terminal.

    ```bash
    python app.py
    ```

4.  **Ask a question:** The program will prompt you to enter a question, and it will provide a concise answer based on the PDF content.

---
### 🙏 Credits

This project is built using the fantastic Langchain framework and `llama-3.1-8b-instant` from Groq.
