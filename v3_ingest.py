#!/usr/bin/env python3
import os
import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import time
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PythonLoader, PyPDFLoader, CSVLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS

# Load environment variables
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME', 'all-MiniLM-L6-v2')
feature = 'aruba_vlan'
source_folder_path = f'/Users/rohanbadiger/Desktop/ModelV3_Nov2024/source_documents_{feature}'
persist_folder_path = f'db_{feature}'
print('Source', source_folder_path)
print('Database', persist_folder_path)
source_directory = os.environ.get('SOURCE_DIRECTORY', source_folder_path)
persist_directory = os.environ.get('PERSIST_DIRECTORY', persist_folder_path)
# Customized chunking
chunk_size = 50 # lines
max_chunk_size = 100 # Max number of lines in a chunk
# Fixed size chunking
Fixed_chunk_size = 600
Fixed_chunk_overlap = 50

def get_python_code_blocks(source_code):
    test_case_script = source_code
    lines = test_case_script.split('\n')

    # get sepcific indexes required
    first_def_index = next((i for i, line in enumerate(lines) if 'def ' in line), None)
    test_index = next((i for i, line in enumerate(lines) if 'def test_' in line), None)

    # fetch indexes of all lines of which next line is empty until first_def_index
    indexes_before_first_def = []
    for i in range(first_def_index):
        if i + 1 < len(lines) and not lines[i + 1].strip() and lines[i].strip():
            indexes_before_first_def.append(i+2)
    indexes_before_first_def.append(first_def_index)

    # fetch indexes of all lines of which next two lines are empty, contain 'step(' in it
    indexes_after_first_def = []
    for i in range(first_def_index, len(lines)):
        if i + 2 < len(lines) and not lines[i + 1].strip() and not lines[i + 2].strip():
            indexes_after_first_def.append(i+3)
        elif 'step(' in lines[i]:
            indexes_after_first_def.append(i)
    indexes_after_first_def.append(len(lines))

    # print(first_def_index, test_index, indexes_before_first_def,indexes_after_first_def)

    # Split the test script into sections based on the indexes until first_def_index
    sections = []
    start = 0
    for i, index in enumerate(indexes_before_first_def):
        while (index-start) < chunk_size and i < len(indexes_before_first_def)-1 and index < first_def_index:
            if indexes_before_first_def[i+1]-start > chunk_size:
                break
            index = indexes_before_first_def[i+1]
            i += 1
        block = '\n'.join(lines[start:index])
        if block.strip():
            block_lines = block.split('\n')
            total_lines = len(block_lines)
            num_sblocks = (total_lines + max_chunk_size - 1) // max_chunk_size
            sblock_size = (total_lines + num_sblocks - 1) // num_sblocks  
            for i in range(num_sblocks):
                sblock = block_lines[i * sblock_size:(i + 1) * sblock_size]
                if sblock:
                    sections.append('\n'.join(sblock))
        start = index
        if i == len(indexes_before_first_def)-1:
            break

    # Split the test script into sections based on the indexes if test_index is not found
    if not test_index or (first_def_index == test_index):
        start = first_def_index
        for i, index in enumerate(indexes_after_first_def):
            while (index-start) < chunk_size and i < len(indexes_after_first_def)-1:
                if indexes_after_first_def[i+1]-start > chunk_size:
                    break
                index = indexes_after_first_def[i+1]
                i += 1
            block = '\n'.join(lines[start:index])
            if block.strip():
                sections.append(block)
            start = index
            if i == len(indexes_after_first_def)-1:
                break
    else: # if test_index is found
        nearest_test_index = max([i for i in indexes_after_first_def if i <= test_index])
        start = first_def_index
        for i, index in enumerate(indexes_after_first_def):
            while (index-start) < chunk_size and i < len(indexes_after_first_def)-1:
                if indexes_after_first_def[i+1]-start > chunk_size or indexes_after_first_def[i] >= nearest_test_index:
                    break
                index = indexes_after_first_def[i+1]
                i += 1
            block = '\n'.join(lines[start:index])
            if block.strip():
                sections.append(block)
            start = index
            if index >= nearest_test_index:
                break

        start = nearest_test_index
        for i, index in enumerate(indexes_after_first_def):
            if index < test_index:
                continue
            while (index-start) < chunk_size and i < len(indexes_after_first_def)-1:
                if indexes_after_first_def[i+1]-start > chunk_size:
                    break
                index = indexes_after_first_def[i+1]
                i += 1
            block = '\n'.join(lines[start:index])
            if block.strip():
                sections.append(block)
            start = index
            if i == len(indexes_after_first_def)-1:
                break

    return sections

def split_code_by_structure(source_code, document):
    """
    Split the source code into logical blocks and retain metadata for tracking.
    """
    blocks = get_python_code_blocks(source_code)
    lines = source_code.splitlines()

    chunks = []
    for i, block in enumerate(blocks):
        chunk_with_metadata = Document(page_content=block, metadata={'source': document.metadata["source"], 'chunk_no': i+1})
        chunks.append(chunk_with_metadata)

    return chunks

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")    
    loader = DirectoryLoader(source_directory, glob="**/*.py", use_multithreading=True, loader_cls=PythonLoader, show_progress=True)
    documents = loader.load()
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    # return documents

    # Chunking based on Fixed size
    # text_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # texts = text_splitter.split_documents(documents)
    # print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")

    # Chunking based on python structure of Test scripts
    texts = []
    for document in documents:
        source = document.metadata["source"]
        file_name = source.rsplit('/', 1)[-1]  # Get only the file name
        # print(file_name)
        if file_name.startswith('test_'):
            source_code = document.page_content  # Extract source code
            python_chunks = split_code_by_structure(source_code, document)
            texts.extend(python_chunks)
            # print(f"{source} split into {len(python_chunks)} chunks based on its Python structure.")
        else:
            chunks = []
            chunk = Document(page_content=document.page_content, metadata={'source': document.metadata["source"], 'chunk_no': 1})
            chunks.append(chunk)
            texts.extend(chunks)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} lines each)")

    return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False

def main():
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    start = time.time()
    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        print(f"Creating embeddings. May take some minutes...")

        batch_size = 1000
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            print(f"Adding batch {i//batch_size + 1}/{len(texts)//batch_size + 1} to db...")
            db.add_documents(batch)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        
        batch_size = 1000
        for i in range(0, len(texts), batch_size):
            cur_time = time.strftime("%H:%M:%S", time.localtime())
            batch = texts[i:i+batch_size]
            print(f"Adding batch {i//batch_size + 1}/{len(texts)//batch_size + 1} to db... {cur_time}")
            if i == 0:
                db = Chroma.from_documents(batch, embeddings, persist_directory=persist_directory)
            else:
                db.add_documents(batch)
    db.persist()
    db = None
    end = time.time()
    time_taken = end - start
    hours, min_rem = divmod(time_taken, 3600)
    minutes, seconds = divmod(min_rem, 60)
    time_taken = f"{int(hours)}h:{int(minutes):02}m:{int(seconds):02}s"
    print(f"Ingestion complete! You can now run query.py to query your documents")
    print(f"Total Elapsed time: {time_taken} ")

if __name__ == "__main__":
    main()
