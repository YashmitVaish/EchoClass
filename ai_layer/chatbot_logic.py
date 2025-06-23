
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import retrieval_qa   
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv("ai_layer/keys.env")
openai_api_key = os.getenv("OPENAI_API")
api = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_SUMM")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

embed = OpenAIEmbeddings(
    model='text-embedding-3-small',
    openai_api_key=openai_api_key
)

from langchain_pinecone import PineconeVectorStore

pinecone_vectorstore = PineconeVectorStore(
    index_name= "rag-search-engine", 
    embedding=embed, 
    text_key="chunk_text"
)

query = "test-query"

documents = pinecone_vectorstore.similarity_search(
    query=query,
    k=3  
)


template=(
  "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."
  "Question: {question}"
  "Context: {context}"
  "Answer:"
)
prompt = PromptTemplate(input_variables=["question", "context"], template=template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {
      "context": pinecone_vectorstore.as_retriever() | format_docs,
      "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

qa_chain.invoke(query)




