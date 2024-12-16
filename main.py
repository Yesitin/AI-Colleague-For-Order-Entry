from langchain.document_loaders import PyPDFLoader
import streamlit as st
import os
from dotenv import load_dotenv
from functions import *

st.set_page_config(layout="wide")   # for streamlit full screen


# to load api key from .env file

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    st.error("API key is missing")


st.title("Order into Database")




# Upload Button to upload PDF file and creating a temporary temp.pdf file

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Loading PDF..."):
        with open("data/temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())


        # load PDF
        loader = PyPDFLoader("data/temp.pdf")
        pages = loader.load()


    embedding_function = get_embedding_function()



# Execute Button to run whole process of creating chunks, creating embeddings, creating vectorstore database and parsing data from document

    if st.button("Execute"):
        with st.spinner("Processing..."):
            st.session_state.vectorstore = create_vectorstore_db(pages, embedding_function, filename=uploaded_file.name)

            st.session_state.df = query_vectorstore(st.session_state.vectorstore)
            st.success("Order created")

            st.dataframe(st.session_state.df.T, width=2000, height=465)




# Save to DB Button to save parsed data into a SQL database
# Query Button to query to database (no limit, everything will be shown)

    if "df" in st.session_state and st.session_state.df is not None:

        if st.button("Save to Database"):
            save_data_sql(st.session_state.df)
            st.success("Data saved into Database")

        if st.button("Query orders"):
            query_data = query_sqldatabase()
            st.write(query_data)