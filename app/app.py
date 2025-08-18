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


import spacy
nlp = spacy.load("en_core_web_sm")
all_chunks_entities = []
for doc in docs:  # each doc is one chunk
    text = doc.page_content
    spacy_doc = nlp(text)
    chunk_entities = []
    for ent in spacy_doc.ents:
        chunk_entities.append((ent.text, ent.label_))
    all_chunks_entities.append(chunk_entities)
chunks_entities = all_chunks_entities
# print(chunks_entities)


from dotenv import load_dotenv
import os
from neo4j import GraphDatabase
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def create_entity_node(tx, entity_name, entity_label):
    query = (
        f"MERGE (e:{entity_label} {{name: $entity_name}})"
    )
    tx.run(query, entity_name=entity_name)



def create_relationship(tx, entity1, label1, entity2, label2):
    query = (
        f"MATCH (a:{label1} {{name: $entity1}}), (b:{label2} {{name: $entity2}}) "
        f"MERGE (a)-[:RELATED_TO]->(b)"
    )
    tx.run(query, entity1=entity1, entity2=entity2)



def ingest_entities_to_neo4j(driver, chunks_entities):
    with driver.session() as session:
        for entities in chunks_entities:
            # Create nodes for all entities in the chunk
            for name, label in entities:
                session.write_transaction(create_entity_node, name, label)
            
            # Create relationships between every pair of entities in the chunk
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    name1, label1 = entities[i]
                    name2, label2 = entities[j]
                    session.write_transaction(create_relationship, name1, label1, name2, label2)
    print("Entities and relationships ingested into Neo4j!")


# chunks_entities = all_chunks_entities
# ingest_entities_to_neo4j(driver, chunks_entities)


from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
load_dotenv()
llm_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=llm_api_key,model_name="llama-3.1-8b-instant", temperature=0.5)


answer_template = """
You are a helpful assistant.
Given the following context extracted from documents and a knowledge graph:
{context}
Answer the question:
{question}
"""


from langchain.prompts import PromptTemplate
answer_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=answer_template,
)


from langchain.chains import LLMChain
answer_chain = LLMChain(llm=llm, prompt=answer_prompt)


def traverse_graph(driver, entities, max_hops=1):
    with driver.session() as session:
        results = []
        for name, label in entities:
            query = f"""
            MATCH p = (n:{label} {{name: $name}})-[r*1..{max_hops}]-(m)
            RETURN DISTINCT n.name AS source, 
                   labels(n) AS source_labels,
                   type(r[0]) AS rel_type,
                   m.name AS target,
                   labels(m) AS target_labels
            LIMIT 25
            """
            records = session.run(query, name=name)
            for record in records:
                results.append({
                    "source": record["source"],
                    "source_labels": record["source_labels"],
                    "relationship": record["rel_type"],
                    "target": record["target"],
                    "target_labels": record["target_labels"]
                })
        return results

    

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]
    

from langchain.schema import Document    
def answer_query(query, vectorstore, driver, top_k=3):
    # 1. Embed and retrieve top-k relevant chunks
    docs: list[Document] = vectorstore.similarity_search(query, k=top_k)
    
    # 2. Extract entities from query + retrieved chunks
    query_entities = extract_entities(query)
    chunk_entities = []
    for doc in docs:
        chunk_entities.extend(extract_entities(doc.page_content))
    
    # Combine and dedupe entities (simple way)
    all_entities = list({(e[0], e[1]) for e in query_entities + chunk_entities})
    
    # 3. Traverse Neo4j graph around those entities to get connected info
    graph_info = traverse_graph(driver, all_entities)
    
    # 4. Assemble context string from text chunks + graph info
    context_texts = [doc.page_content for doc in docs]
    
    graph_texts = []
    for edge in graph_info:
        s_lbl = ",".join(edge["source_labels"])
        t_lbl = ",".join(edge["target_labels"])
        rel = edge["relationship"] or "RELATED_TO"
        graph_texts.append(f"{edge['source']} ({s_lbl}) -[{rel}]-> {edge['target']} ({t_lbl})")
    
    combined_context = "\n\n".join(context_texts) + "\n\nGraph relationships:\n" + "\n".join(graph_texts)
    
    # 5. Use LLM chain to answer with combined context
    result = answer_chain.run(context=combined_context, question=query)
    
    return result


print("Ready for questions! Type 'exit' to quit.")
while True:
    user_q = input("Question: ")
    if user_q.lower() in ["exit", "quit"]:
        break
    answer = answer_query(user_q, vectorstore, driver)
    print("Answer:\n", answer)