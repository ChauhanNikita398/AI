# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 16:32:20 2023

@author: Arnav
"""

import sys
import os
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma
#from langchain.vectorstores import Pinecone
from langchain.vectorstores import Qdrant
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate

#import pinecone

os.environ["OPENAI_API_KEY"] = "sk-xxxxx"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_xxxxxx"
PINECONE_API_KEY="xxxxxx"
PINECONE_ENV="xxxxxx"

documents = []
for file in os.listdir("C:\\Users\\nikita\\Documents\\hackathon\\content-summ\\src\\input"):
    if file.endswith(".pdf"):
        pdf_path = "C:\\Users\\nikita\\Documents\\hackathon\\content-summ\\src\\input\\" + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        doc_path = "C:\\Users\\nikita\\Documents\\hackathon\\content-summ\\src\\input\\" + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        text_path = "C:\\Users\\nikita\\Documents\\hackathon\\content-summ\\src\\inputs\\" + file
        loader = TextLoader(text_path)
        documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
docs = text_splitter.split_documents(documents)

#vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data")
#vectordb.persist()

#pinecone.init(
   # api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    #environment=PINECONE_ENV,  # next to api key in console
#)

embedding=OpenAIEmbeddings()
index_name="test-index"
#vectordb = Pinecone.from_documents(docs, embedding=HuggingFaceEmbeddings(), index_name=index_name)

vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data")
vectordb.persist()


# qdrant = Qdrant.from_documents(
#     docs,
#     embedding,
#     path="D:\\Technical\\PythonProjects\\suj-content-summ\\local_qdrant",
#     collection_name="my_documents",
# )

pdf_qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo"),
    #HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64}),
    vectordb.as_retriever(search_kwargs={'k': 6}),
    #qdrant.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=False
)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

chat_history = []
print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome to the DocBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')
while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = pdf_qa(
        {"question": query, "chat_history": chat_history})
    print(f"{white}Answer: " + result["answer"])
    chat_history.append((query, result["answer"]))
