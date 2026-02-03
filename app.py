import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

@st.cache_resource
def setup():
    docs = PyPDFDirectoryLoader("documents/").load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    return Chroma.from_documents(chunks, OpenAIEmbeddings()).as_retriever()

retriever = setup()

prompt = ChatPromptTemplate.from_template("Contexto: {context}\n\nPregunta: {question}")
chain = (
    {"context": retriever | (lambda docs: "\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-3.5-turbo")
    | StrOutputParser()
)

pregunta = st.text_input("Pregunta")
if pregunta:
    st.write(chain.invoke(pregunta))