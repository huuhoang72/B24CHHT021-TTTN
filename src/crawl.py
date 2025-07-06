import os
import re
import json
from langchain_community.document_loaders import (
    RecursiveUrlLoader, WebBaseLoader,
    PyPDFLoader, UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

def crawl_web(url_data):
    loader = RecursiveUrlLoader(url=url_data, extractor=bs4_extractor, max_depth=4)
    docs = loader.load()
    print('length:', len(docs))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    all_splits = text_splitter.split_documents(docs)
    print('length_all_splits:', len(all_splits))
    return all_splits

def web_base_loader(url_data):
    loader = WebBaseLoader(url_data)
    docs = loader.load()
    print('length:', len(docs))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    all_splits = text_splitter.split_documents(docs)
    return all_splits

def load_pdf_file(filepath):
    loader = PyPDFLoader(filepath)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    all_splits = text_splitter.split_documents(docs)
    print('Loaded PDF:', filepath, '->', len(all_splits), 'chunks')
    return all_splits

def load_docx_file(filepath):
    loader = UnstructuredWordDocumentLoader(filepath)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    all_splits = text_splitter.split_documents(docs)
    print('Loaded DOCX:', filepath, '->', len(all_splits), 'chunks')
    return all_splits

def save_data_locally(documents, filename, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    data_to_save = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in documents]
    with open(file_path, 'w') as file:
        json.dump(data_to_save, file, indent=4)
    print(f'Data saved to {file_path}')

def main():
    # === CÁCH 1: Crawl Web ===
    data_web = crawl_web('https://www.stack-ai.com/docs')
    save_data_locally(data_web, 'stack_web.json', 'data')

    # === CÁCH 2: Đọc file PDF ===
    data_pdf = load_pdf_file('data/sample.pdf')  # <-- sửa path này theo file bạn có
    save_data_locally(data_pdf, 'sample_pdf.json', 'data')

    # === CÁCH 3: Đọc file Word (.docx) ===
    data_docx = load_docx_file('data/sample.docx')  # <-- sửa path này theo file bạn có
    save_data_locally(data_docx, 'sample_docx.json', 'data')

if __name__ == "__main__":
    main()
