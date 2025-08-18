# RAG Guest Lecture
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/Rudalph/RAG_Guest_Lecture/tree/main)

This repository contains the code and resources for a guest lecture on Retrieval-Augmented Generation (RAG). It includes three distinct implementations of a question-answering system that uses a local PDF document as its knowledge base.

*   **Basic RAG (`app.py`):** A fundamental RAG pipeline that loads a PDF, chunks it, creates embeddings, and uses an in-memory vector store (Chroma) to answer questions.
*   **Persistent RAG (`main/main.py`):** An example that demonstrates creating and loading from a persistent Chroma vector database on disk.
*   **Knowledge Graph RAG (`app/app.py`):** An advanced pipeline that combines vector retrieval with a knowledge graph. It uses spaCy for named entity recognition and Neo4j to build and query a graph of relationships, providing a richer context to the LLM.

## Repository Structure

*   `app.py`: The primary script for the basic RAG implementation.
*   `main/main.py`: Demonstrates using a persistent ChromaDB vector store. The repository includes a pre-built `chroma_db` for this example.
*   `app/app.py`: Implements the advanced Knowledge Graph RAG (KG-RAG) pipeline.
*   `data/`: This directory is intended to hold the PDF documents you want to use as the knowledge source.
*   `requirements.txt`: Contains the necessary Python packages for the basic and persistent RAG examples.

## Setup and Usage

### 1. Clone the Repository
```bash
git clone https://github.com/Rudalph/RAG_Guest_Lecture.git
cd RAG_Guest_Lecture
```

### 2. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

```bash
# Install virtualenv if you haven't already
pip install virtualenv

# Create a virtual environment
python -m venv env

# Activate the virtual environment
# On Windows
.\env\Scripts\activate
# On macOS/Linux
source env/bin/activate
```

### 3. Install Dependencies
Install the required packages from `requirements.txt`.

```bash
pip install -r requirements.txt
```

For the advanced **Knowledge Graph RAG** example (`app/app.py`), you will also need to install `spacy`, its English model, and the `neo4j` driver.

```bash
pip install spacy neo4j
python -m spacy download en_core_web_sm
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory of the project. This file will store your API keys and database credentials.

Your `.env` file should look like this:
```
GROQ_API_KEY="your_groq_api_key"

# Add these for the Knowledge Graph RAG example (app/app.py)
NEO4J_URI="your_neo4j_bolt_uri"
NEO4J_USER="your_neo4j_username"
NEO4J_PASSWORD="your_neo4j_password"
```

### 5. Add Your Data
Place the PDF file you wish to query inside the `data/` directory. The scripts are currently configured to use `data/Blockchain_Guest_Lecture_Notes.pdf`. If you use a different file, be sure to update the filename in the corresponding Python script.

### 6. Running the Examples

#### Basic RAG (`app.py`)
This script creates an in-memory vector store each time it runs.
```bash
python app.py
```
The script will then prompt you to enter a question.

#### Persistent RAG (`main/main.py`)
This script loads a pre-existing vector store from the `main/chroma_db` directory.
```bash
cd main
python main.py
```
The script will then prompt you to enter a question.

#### Knowledge Graph RAG (`app/app.py`)
This script requires a running Neo4j instance with the correct credentials in your `.env` file. The script first ingests entities and relationships from the PDF into Neo4j and then uses both the graph and vector search to answer questions.

**Note:** You may need to uncomment the `ingest_entities_to_neo4j(driver, chunks_entities)` line on the first run to populate your Neo4j database.

```bash
cd app
python app.py
```

After an initial processing step, the script will be ready to take your questions. Type `exit` to quit.
