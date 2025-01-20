#!/usr/bin/env python3
import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PythonLoader, PyPDFLoader, CSVLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import warnings
from time import sleep

warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub')

# Load environment variables
db_source = '/Users/rohanbadiger/Desktop/ModelV3_Nov2024/db_aruba_vlan'
PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY', db_source)
EMBEDDINGS_MODEL_NAME = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
TARGET_SOURCE_CHUNKS = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

template = """
Give Answer

CONTEXT: {context}
</s>
{question}
</s>
"""

PROMPT = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

def main():
    # Initialize Streamlit
    st.set_page_config(layout="wide")
    st.title("Halon CX Switching Automation - Chatbot")
    st.markdown("---")

    # Define three columns
    col1, col2, col3 = st.columns(3)

    # Add checkboxes to the columns
    hide_source = col1.checkbox("Hide source documents", value=False, key="hide_source")

    os.environ["GOOGLE_API_KEY"] = ''
    
    uploaded_files = st.file_uploader("Upload Python files", type=["py"], accept_multiple_files=True)
    files_content = {}

    if uploaded_files:
        for file in uploaded_files:
            # Read content of each Python file
            file_content = file.read().decode("utf-8")
            files_content[file.name] = file_content  # Store content in a dictionary

        st.success(f"File {file.name} saved successfully!")
        st.warning('Ingestion Complete!')
        # Iterate over the files and print their content
        # for filename, content in files_content.items():
        #     st.write(f"### File: {filename}")
        #     st.code(content, language="python")

    user_input = st.chat_input("Enter a query:")
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if user_input:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
        db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)

        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
        retriever = db.as_retriever(search_kwargs={"k": TARGET_SOURCE_CHUNKS})

        # Initialize the QA system
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=not hide_source,
            chain_type_kwargs={"prompt": PROMPT}
        )

        # Process user query
        res = qa(user_input)
        answer, docs = res['result'], [] if hide_source else res['source_documents']
        st.session_state.chat_history.append({"user": user_input, "answer": answer})

        # Display chat history
        for idx, chat in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(chat['user'])
            with st.chat_message("assistant"):
                st.write(chat['answer'])
        
        # Display Source documents
        if not hide_source:
            st.write("**Source Documents:**")
            for index, document in enumerate(docs, start=1):
                source_label = f"**Source {index}:**"
                #st.write(source_label)
                source_path = document.metadata['source']
                path = source_path.rsplit('/')
                st.write(f"{source_label} {path[-2]}->{path[-1]}")
                #st.write(f"{source_label} {source_path}")
                st.code(document.page_content, language="python")

if __name__ == "__main__":
    main()
