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
DOCUMENTS_PATH = "Documents/"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
LLM_MODEL = "gpt-3.5-turbo"

@st.cache_resource
def initialize_rag_system():
    """
    Inicializa el sistema RAG cargando documentos, dividiéndolos en fragmentos,
    y creando un almacén vectorial con recuperador.
    Retorna: Retriever: Un objeto recuperador para búsqueda semántica
    """

    loader = PyPDFDirectoryLoader(DOCUMENTS_PATH)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter( chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents( documents=chunks, embedding=OpenAIEmbeddings())
    
    return vectorstore.as_retriever()

def format_retrieved_documents(docs):
    """Formatea los documentos recuperados en una cadena de contexto única."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever):
    """
    Crea la cadena RAG que combina recuperación y generación.
    Documentación: https://python.langchain.com/docs/use_cases/question_answering/
    """
    prompt_template = ChatPromptTemplate.from_template("Contexto: {context}\n\nPregunta: {question}\n\nRespuesta:")
    chain = (
        {"context": retriever | format_retrieved_documents, "question": RunnablePassthrough()}
        |prompt_template| ChatOpenAI(model=LLM_MODEL, temperature=0)| StrOutputParser())
    
    return chain

#Inicio
retriever = initialize_rag_system()
rag_chain = create_rag_chain(retriever)

st.title("Sistema RAG")
st.markdown("Haz preguntas sobre los documentos PDF cargados en el sistema.")


user_question = st.text_input("Escribe tu pregunta:",placeholder="Ej: ¿Qué información contiene este documento?")
if user_question:
    with st.spinner("Buscando información y generando respuesta..."):
        answer = rag_chain.invoke(user_question)
        st.write("Respuesta:")
        st.write(answer)