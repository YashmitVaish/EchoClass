from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
import dotenv
import os

dotenv.load_dotenv("ai_layer/keys.env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
embed = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

def search(text: str) -> list:
    vector = embed.embed_query(text)

    index = pc.Index(index_name="rag-search-engine")  

    results = index.query(
        vector=vector,
        top_k=3,
        namespace="pdf-search",  
        include_metadata=True
    )

    return [match['metadata'].get('lec_id') for match in results['matches']]
