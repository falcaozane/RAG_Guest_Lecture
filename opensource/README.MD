## Kaggle Link : 

https://www.kaggle.com/code/zanefalcao/rag-using-open-source-llm-s

```
bash

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

```
python

docs_mmr = langchain_chroma.max_marginal_relevance_search(question, k=5)

```

2. Similarity Search
Similarity search finds the documents whose vector embeddings are closest to the query's vector embedding in a high-dimensional space. It is a straightforward approach that prioritizes the most relevant content, but it has one key drawback: if your vector store contains many documents covering the same topic, the search may return redundant information. 
Under the hood: The search relies on a distance metric, most commonly cosine similarity, which measures the angle between the document and query vectors. A smaller angle means greater similarity.
Example scenario: If your dataset contains three paragraphs about a specific type of AI, like "generative AI," a similarity search for the query "What is generative AI?" will likely return all three paragraphs because they are mathematically the most similar. 

```
python
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




