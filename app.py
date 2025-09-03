# Importing necessary libraries
import os
from dotenv import load_dotenv
import streamlit as st
import pickle
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
import nest_asyncio

nest_asyncio.apply()
load_dotenv()

st.title("Article research tool")
st.sidebar.title("Enter URL's")
main_placeholder = st.empty()

try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n",".",","],
        chunk_size=1000,
        chunk_overlap=100
    )
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():
        urls.append(url)

process_button_clicked = st.sidebar.button("Read URL's")

if process_button_clicked:

    main_placeholder.text("Loading data from urls...")
    try:
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        if not data:
            st. error("No content")
            st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()


    main_placeholder.text("Splitting data into chunks...")
    try:
        docs = text_splitter.split_documents(data)
        if not docs:
            st.error("No split")
            st.stop()
    except Exception as e:
        st.error(f"Error splitting data: {e}")
        st.stop()

    main_placeholder.text("Embedding and storing vectors...")  
    try:
        vectors = FAISS.from_documents(docs,embeddings)
        vectors.save_local("vector_data")
    except Exception as e:
        st.error(f"Error embedding data: {e}")
        st.stop()

query = main_placeholder.text_input("Question:")
if query:
    if os.path.exists("vector_data/index.pkl"):
        vectorstore = FAISS.load_local("vector_data",embeddings= embeddings,allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm = llm, retriever = vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs = True)
        st.write("Answer:")
        st.write(result["answer"])
    else:
        main_placeholder.text("No data to answer the question")