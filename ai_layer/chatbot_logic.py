
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv("ai_layer/keys.env")
openai_api_key = os.getenv("OPENAI_API")
api = os.getenv("PINECONE_API_KEY")

embed = OpenAIEmbeddings(
    model='text-embedding-3-small',
    openai_api_key=openai_api_key
)

from langchain_pinecone import PineconeVectorStore

pinecone_vectorstore = PineconeVectorStore(
    index_name= "rag-engine", 
    embedding=embed, 
    text_key="text"
)

