## Kaggle Link : 

https://www.kaggle.com/code/zanefalcao/rag-using-open-source-llm-s

```bash

!pip install langchain langchain-community langchain-core transformers sentence_transformers langchain-huggingface pypdf chromadb

```

---

## Document Loaders:

https://medium.com/data-and-beyond/document-loaders-in-langchain-f23d3ce70d66

---


## Text Splitters

| **Splitter**                       | **Description**                                                                                                                                   | **Use Case**                                                                                              |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **RecursiveCharacterTextSplitter** | Recommended general-purpose splitter. Recursively splits based on characters like `["\n\n", "\n", " ", ""]`, moving from larger to smaller units. | Best for splitting generic text (articles, reports) while maintaining context.                            |
| **CharacterTextSplitter**          | Splits text based on a single, fixed character. Does not preserve larger document structures.                                                     | Good for unstructured text (FAQs, chatbot prompts) where consistent chunk size matters more than context. |
| **TokenTextSplitter**              | Splits text based on token limits (how LLMs interpret text). Ensures chunks fit within a model’s context window.                                  | Ideal for tasks sensitive to token limits, e.g., optimizing input for GPT models.                         |
| **MarkdownTextSplitter**           | Splits Markdown documents while respecting headers, code blocks, and syntax.                                                                      | Use for Markdown docs, technical reports, blogs to preserve hierarchy.                                    |
| **HTMLTextSplitter**               | Preserves HTML structure while splitting web content.                                                                                             | Best for extracting content from HTML pages for analysis or processing.                                   |
| **NLTKTextSplitter**               | Uses NLTK to split text based on linguistic features (sentences, paragraphs).                                                                     | Suitable for well-structured text where linguistic coherence matters.                                     |
| **PythonCodeTextSplitter**         | Optimized for Python code; respects classes and function definitions.                                                                             | Ensures code structure integrity when processing Python files.                                            |
| **LatexTextSplitter**              | Splits LaTeX documents while respecting LaTeX commands and environments.                                                                          | Designed for academic/scientific papers written in LaTeX.                                                 |

***
---

## Prompt templates :

https://medium.com/data-and-beyond/prompts-in-langchain-with-examples-0fb30be66b81
---


## ChromaDB 

1. Maximal Marginal Relevance (MMR) search
Instead of just returning the most relevant documents, MMR finds documents that are both relevant to the query and diverse among themselves. This helps prevent your results from being filled with near-duplicate information. 
When to use it: Use MMR when you want to avoid redundancy in your search results. For example, if you are searching for "AI applications" and want to retrieve documents covering different aspects like "generative AI" and "computer vision," MMR will help you get a wider range of results.

```python

docs_mmr = langchain_chroma.max_marginal_relevance_search(question, k=5)

```

2. Similarity Search
Similarity search finds the documents whose vector embeddings are closest to the query's vector embedding in a high-dimensional space. It is a straightforward approach that prioritizes the most relevant content, but it has one key drawback: if your vector store contains many documents covering the same topic, the search may return redundant information. 
Under the hood: The search relies on a distance metric, most commonly cosine similarity, which measures the angle between the document and query vectors. A smaller angle means greater similarity.
Example scenario: If your dataset contains three paragraphs about a specific type of AI, like "generative AI," a similarity search for the query "What is generative AI?" will likely return all three paragraphs because they are mathematically the most similar. 

```python

docs_pdf = langchain_chroma.similarity_search(question,k=5)
```
---


| **Use Case**          | **When to use `similarity_search`**                                                              | **When to use `MMR` (Maximal Marginal Relevance)**                                                                   |
| --------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| **Summarization**     | Generate a short, focused summary based on the most relevant information.                        | Generate a comprehensive summary that covers a broader range of topics related to the query.                         |
| **Conversational AI** | User asks a very specific, unambiguous question with a single correct answer.                    | User’s query is broad or complex; ensure the chatbot considers all relevant facets before responding.                |
| **Product Search**    | Return the most popular and highly rated models first (e.g., “running shoes”).                   | Show a variety of options to cover different aspects (e.g., “fresh red tomatoes,” “canned diced,” “organic cherry”). |
| **General Search**    | Find documents that are the closest semantic match, even if they are very similar to each other. | Explore different aspects of a topic without being overwhelmed by redundant information.                             |

***

---

https://docs.langchain.com/oss/python/langchain/install

https://docs.langchain.com/oss/python/integrations/providers/overview

https://docs.langchain.com/oss/python/integrations/providers/chroma



This command installs the complete "tech stack" required to build your local RAG system. It’s helpful to group them into three categories: **The Orchestrator** (LangChain), **The Brains** (Hugging Face), and **The Data Layer** (PDFs & Database).

Here is the breakdown of what each package does and why you need it for your specific code.

### **1. The Orchestrator (LangChain Family)**
LangChain is the framework that glues everything together. It used to be one giant package, but it was recently split into smaller pieces to be cleaner and faster.

* **`langchain-core`**: The absolute basics. It defines the standard "blueprints" for how things should look—like what a "Prompt Template" is, or what a "Document" looks like. It ensures that a ChatBot and a QA Chain speak the same language.
* **`langchain-community`**: This contains all the **integrations**. Since LangChain connects to thousands of tools (Google, OpenAI, Wikipedia, etc.), they moved all those specific connections here to keep the main package light. Your `PyPDFLoader` comes from here.
* **`langchain`**: The main library. It contains the logic for **"Chains"** (linking steps together) and **"Retrieval"** (finding data). It uses `core` for structure and `community` for tools to build your actual application.

### **2. The Brains (Hugging Face Family)**
These packages allow you to run the AI models locally on your machine instead of using an API.

* **`transformers`**: The most famous AI library in the world (by Hugging Face). It allows you to download and run the actual Large Language Models (like `Falcon-1B` or `Llama-3`). It handles the heavy lifting of loading the model weights into your RAM.
* **`sentence_transformers`**: A specialized library designed *specifically* to turn text into numbers (embeddings). While `transformers` is good for *generating* text, `sentence_transformers` is optimized for *understanding* similarity between sentences. You use this to turn your PDF chunks into vectors.
* **`langchain-huggingface`**: The bridge. LangChain speaks "chains" and Hugging Face speaks "tensors." This package acts as a translator so you can plug a Hugging Face model directly into a LangChain pipeline without writing complex custom code.

### **3. The Data Layer (Storage & Ingestion)**
These tools handle your specific data needs.

* **`pypdf`**: A lightweight tool to read PDF files. It extracts the raw text from your uploaded document so Python can actually read it (since Python can't read `.pdf` binary files directly).
* **`chromadb`**: The **Vector Database**. Normal databases (like SQL) store rows and columns. Chroma is built to store **Vectors** (lists of numbers). It is optimized to answer questions like: *"Which 5 rows are mathematically closest to this query?"* It acts as the long-term memory for your RAG system.

### **Summary: How they work together in your code**
1.  **`pypdf`** reads the file.
2.  **`langchain`** splits the text into chunks.
3.  **`sentence_transformers`** turns chunks into math (vectors).
4.  **`chromadb`** saves those vectors.
5.  **`transformers`** loads the Falcon AI model.
6.  **`langchain-core`** & **`langchain-community`** build the pipeline that connects the User Question -> Database -> AI Model -> Answer.


When you call embeddings = HuggingFaceEmbeddings(), the following happens:
Downloads a model: LangChain uses the sentence-transformers library to automatically download a default embedding model from the Hugging Face Hub. 
By default, it uses a widely-used, high-performance model such as sentence-transformers/all-mpnet-base-v2


```python
from langchain_huggingface import HuggingFaceEmbeddings

# Specify a different model and force it to run on the CPU
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2", # A smaller, faster model
    model_kwargs={'device': 'cpu'} # Run on CPU instead of default
)

```

For a list of compatible models and their rankings, you can check the Massive Text Embedding Benchmark (MTEB) Leaderboard on Hugging Face. 

```python

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
sentences = [
    "Hugging Face makes NLP easy.",
    "Transformers are powerful for deep learning."
]
embeddings = model.encode(sentences)
print("Embedding shape:", embeddings.shape)
```
---




