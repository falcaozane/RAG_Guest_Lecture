from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("data/Blockchain_Guest_Lecture_Notes.pdf")
data = loader.load()
# print(data)


from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
docs = text_splitter.split_documents(data)
# print(docs)
                

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


from langchain_chroma import Chroma
vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings
)


retriever = vectorstore.as_retriever(search_type="similarity")



from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
load_dotenv()
llm_api_key = os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=llm_api_key,model_name="llama-3.1-8b-instant", temperature=0.5)


user_query = input("Enter your question: ")


from langchain_core.prompts import ChatPromptTemplate
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])



from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
if user_query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": user_query})

    print(response["answer"])

                
                
                