# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 22:02:37 2026

@author: sanhi
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDFs
loaders = [
    PyPDFLoader(r"C:\Users\sanhi\Downloads\ReviewPaper_Fusion.pdf"),
    PyPDFLoader(r"C:\Users\sanhi\Downloads\Radar and Camera Early Fusion for Vehicle Detection in Advanced Driver Assistance Systems.pdf"),
    PyPDFLoader(r"C:\Users\sanhi\Downloads\Dual Perspective Fusion Transformer.pdf"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
#load.loader=loads pages...
#docs.extend=adds all pages from all the pdfs loaded
print(f"Loaded {len(docs)} pages total")#total no of pages

#splitting the info in groups of 500 charactors with 50 characters overlapping
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"Total chunks: {len(chunks)}")

#Embed all chucks and store in Chroma DB
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text")
for chunk in chunks:
    chunk.page_content = chunk.page_content.encode('utf-8', 'ignore').decode('utf-8')#ifnore math symbols causing error for embedding
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./fusion_knowledge_base"
)
print("Knowledge base built and saved.")