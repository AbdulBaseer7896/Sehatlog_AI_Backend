# # from langchain.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain.embeddings import HuggingFaceEmbeddings
# # from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings



# #Extract Data From the PDF File
# def load_pdf_file(data):
#     loader= DirectoryLoader(data,
#                             glob="*.pdf",
#                             loader_cls=PyPDFLoader)

#     documents=loader.load()

#     return documents



# #Split the Data into Text Chunks
# def text_split(extracted_data):
#     text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
#     text_chunks=text_splitter.split_documents(extracted_data)
#     return text_chunks



# #Download the Embeddings from HuggingFace 
# def download_hugging_face_embeddings():
#     embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  #this model return 384 dimensions
#     return embeddings


# helper.py
import os
import requests
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import  DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class HuggingFaceAPIEmbeddings(Embeddings):
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        self.headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={"inputs": texts, "options": {"wait_for_model": True}}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error getting embeddings: {str(e)}")

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]



def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

# #Extract Data From the PDF File
def load_pdf_file(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)

    documents=loader.load()

    return documents

def download_hugging_face_embeddings():
    return HuggingFaceAPIEmbeddings()