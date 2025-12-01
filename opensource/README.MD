## Kaggle Link : 

https://www.kaggle.com/code/zanefalcao/rag-using-open-source-llm-s

```
bash

!pip install langchain langchain-community langchain-core transformers sentence_transformers langchain-huggingface pypdf chromadb

```


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



