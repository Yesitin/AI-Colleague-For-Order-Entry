from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
import uuid
import os
import pandas as pd
import re
import sqlite3


# to load api key from .env file
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
   


# function to clean file name to prevent errors
def clean_filename(filename):
    new_filename = re.sub(r'\s\(\d+\)', '', filename)
    new_filename = new_filename.replace(' ', '')   
    return new_filename



# function to split document due to token limits of llm's (adjustable)
def create_chunks(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400,                 # max size for each chunk (in characters)
                                                chunk_overlap=60,                  # how many characters overlap betw. chunks
                                                length_function=len,               
                                                separators=["\n\n", "\n", " "])

    chunks = text_splitter.split_documents(pages)

    return chunks



# function to create embeddings so llm can process data
def get_embedding_function():             
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY
    )
    return embeddings



# function to create vector database
def create_vectorstore(chunks, embedding_function, vectorstore_path, filename):

    # Create a list of unique ids for each document based on the content (no duplicates)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
    
    # Ensure that only unique docs with unique ids are kept
    unique_ids = set()
    unique_chunks = []
    
    for chunk, id in zip(chunks, ids):     
        if id not in unique_ids:       
            unique_ids.add(id)
            unique_chunks.append(chunk) 

    # Create a new Chroma database from the documents
    vectorstore = Chroma.from_documents(documents=unique_chunks, 
                                        collection_name=clean_filename(filename),
                                        ids=list(unique_ids),
                                        embedding=embedding_function,
                                        persist_directory = vectorstore_path)
    
    return vectorstore



# function to create vectorstore database
def create_vectorstore_db(pages, embedding_function, filename):
    
    chunks = create_chunks(pages)

    # Create vectorstore
    vectorstore = create_vectorstore(chunks=chunks, 
                                    embedding_function=embedding_function,
                                    vectorstore_path="vectorstore_chroma",
                                    filename=filename)
    
    return vectorstore




# function to load vectorstore
def load_vectorstore(embedding_function, filename):

    vectorstore = Chroma(persist_directory="vectorstore_chroma", 
                         embedding_function=embedding_function,
                         collection_name=clean_filename(filename))

    return vectorstore




# Prompt Template for instructing the Agent
PROMPT_TEMPLATE = """
You are an assistant for extracting information out of documents (transport orders).
Use the following pieces of retrieved context to answer
the question. 

{context}

---

Execute the following request: {question}

"""



# Generate structured responses
class AnswerWithSources(BaseModel):
    """Concise answer to the question."""
    answer: str = Field(description="Answer to question")
    
class ExtractedInfo(BaseModel):
    """Extracted information about the research article"""
    Orderer: AnswerWithSources
    Loading_location: AnswerWithSources
    Unloading_location: AnswerWithSources
    Loading_date: AnswerWithSources
    Loading_time_window: AnswerWithSources
    Unloading_date: AnswerWithSources
    Unloading_time_window: AnswerWithSources
    Goods: AnswerWithSources
    Weight: AnswerWithSources
    ADR: AnswerWithSources
    Loading_number: AnswerWithSources
    Transport_rate: AnswerWithSources



# function to combine several documents into one and seperating them by two breaks (\n)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



# big function to query the vectorstore based on an instruction and creating a output dataframe
def query_vectorstore(vectorstore):

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

    retriever = vectorstore.as_retriever(search_type="similarity")

    # Create prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm.with_structured_output(ExtractedInfo, strict=True)
        )
    
    # Transform response into a dataframe 
    structured_response = rag_chain.invoke("Give me the ordering company, full loading location, full unloading location, loading date, loading time window, unloading date, unloading time window, type of shipment goods, weight, indication of ADR, loading number, transport rate.")
    df = pd.DataFrame([structured_response.dict()])

    answer_row = []

    for col in df.columns:
        answer_row.append(df[col][0]['answer'])

    # Create new dataframe
    df_orderlist = pd.DataFrame([answer_row], columns=df.columns, index=['answer'])
    
    return df_orderlist



# function to save the output df into a sql database
def save_data_sql(df):

    # Create a SQLite connection
    conn = sqlite3.connect("order_list.db")
    cursor = conn.cursor()

    # Creatomg a table with a primary key
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS order_list (
        id_column INTEGER PRIMARY KEY AUTOINCREMENT,
        Orderer TEXT,
        Loading_location TEXT,
        Unloading_location TEXT,
        Loading_date TEXT,
        Loading_time_window TEXT,
        Unloading_date TEXT,
        Unloading_time_window TEXT,
        Goods TEXT,
        Weight TEXT,
        ADR TEXT,
        Loading_number TEXT,
        Transport_rate TEXT
    );
    """)
    conn.commit()

    # Open the SQLite connection
    conn = sqlite3.connect("order_list.db")

    # Save the DataFrame to the SQL database
    df.to_sql("order_list", conn, if_exists="append", index=False)

    conn.close()



# function  to query the sql db
def query_sqldatabase():

    conn = sqlite3.connect("order_list.db")

    query = f"SELECT * FROM order_list;"        # limit of showed entries can be added for e.g. "SELECT * FROM order_list LIMIT 20;"
    result = pd.read_sql(query, conn)

    conn.close()

    return result