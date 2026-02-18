# Final Report: RAG System Implementation

## 1. Team Members

**Team:**
- Serhii
- Marti

**Work Distribution:**
- **Marti**: Created the base structure of the code
- **Serhii**: Expanded and improved the code
- **Both**: Worked together on the code, design decisions, documentation and presentation

## 2. What We Built

### 2.1 Description

We built a RAG system that allows asking questions about PDF documents. The system:

1. Loads PDFs from a folder
2. Divides them into smaller pieces (chunks)
3. Stores them in a vector database
4. Searches for relevant information when you ask a question
5. Generates an answer using GPT-3.5-turbo

The interface is a simple web page made with Streamlit.

### 2.2 Design Decisions

#### PyPDFDirectoryLoader
Automatically loads all PDFs from a folder.

**Link:** https://python.langchain.com/docs/integrations/document_loaders/pdf

#### RecursiveCharacterTextSplitter
- Size: 1500 characters
- Overlap: 300 characters

**Why:** 1500 characters captures complete ideas in literary texts. The overlap prevents losing information.

**Link:** https://python.langchain.com/docs/modules/data_connection/document_transformers/

#### Chroma
Lightweight vector database, easy to use and fast.

**Link:** https://python.langchain.com/docs/integrations/vectorstores/chroma

#### OpenAIEmbeddings
Converts text into vectors to search by meaning, not just by words.

#### GPT-3.5-turbo
Cheaper than GPT-4, fast and works well. Temperature set to 0 for precise answers.

#### Streamlit
Easy to make web interfaces.

**Link:** https://docs.streamlit.io/

#### LangChain Expression Language
Clean code and easy to modify.

**Link:** https://python.langchain.com/docs/use_cases/question_answering/

### 2.3 Problems and Solutions

#### Problem 1: Document path
The code was looking for "documents/" but the folder was "Documents/".

**Solution:** Changed to "Documents/" and used a constant.

#### Problem 2: Chunk size
1000 characters was too small for literary texts.

**Solution:** Increased to 1500 characters with 300 overlap.

### 2.4 Code Structure

The code has:
1. Configuration (constants)
2. Initialization (loads and processes documents)
3. Helper functions (formatting)
4. RAG chain creation
5. User interface (Streamlit)

### 2.5 Team Contributions

**Marti:**
- Created the base structure of the code
- Implemented the main functions
- Configured the initial system

**Serhii:**
- Expanded and improved the code
- Optimized chunk size
- Improved code structure
- Tested the system with different documents

**Both:**
- Worked together on the code
- Made design decisions
- Wrote documentation
- Prepared the presentation

## 4. Conclusion

We successfully built a working RAG system. The system processes PDFs, creates a vector database and generates answers based on the documents.

**What we learned:**
- How RAG systems work
- How to divide documents into chunks
- How to do semantic search
- How to use LangChain and Streamlit

**Future improvements:**
- Support for more formats (Word, TXT)
- Improve search
- Show answer sources
