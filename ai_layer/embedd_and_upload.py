from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os 
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv("ai_layer/keys.env")
PINECONE_API = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API")
pc = Pinecone(api_key=PINECONE_API)

index_name = "rag-search-engine"
namespace_name = "pdf-search"

index = pc.Index(name=index_name)

embed = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
vector_store = PineconeVectorStore(index=index, embedding=embed)

def get_last_index(index_name:str,namespace_name:str):
    cursor = pc.Index(index_name)
    records = list(cursor.list(namespace = namespace_name))  
    a = []
    b = []
    for item in records:
        items = list(map(lambda x : int(x), item))
        a.append(items)
        for i in a:
            b.append(max(i))
    try:
        return(max(b))
    except:
        return 0

def format_chunks(record: list, last_index: int, lec_id: str):
    return list(map(
        lambda i: {
            "_id": f"doc_{i}",
            "chunk_text": record[i - last_index - 1],
            "metadata": {
                "lec_id": lec_id
            }
        },
        range(last_index + 1, last_index + len(record) + 1)
    ))

def upload(chunks:list,lec_id) -> None:
    documents = format_chunks(chunks,get_last_index(index_name,namespace_name),lec_id)
    vectors = []
    for doc in documents:
        vector = embed.embed_query(doc["chunk_text"])
        vectors.append({
            "id": doc["id"],
            "values": vector,
            "metadata": doc["metadata"]
        })
        
    index.upsert(vectors=vectors)

