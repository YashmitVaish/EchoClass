'''
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

'''

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
import dotenv
import re
import string

dotenv.load_dotenv("ai_layer/keys.env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


def chunk_generator(input_string,chunk_size=1700,chunk_overlap=200):
        splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
        )   
        
        chunks = splitter.split_text(input_string)

        clean_chunks = clean_chunks_list(chunks)
        return clean_chunks


def clean_chunk(text: str) -> str:
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Remove Unicode blocks used in OCR box drawing and ligatures
    text = re.sub(r'[\u2580-\u259F\uFB00-\uFB4F]', '', text)

    # Collapse symbol spam (3+ punctuation characters in a row)
    text = re.sub(r'[' + re.escape(string.punctuation) + r']{3,}', ' ', text)

    # Collapse all whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Heuristics to filter out garbage
    stripped = text.replace(" ", "")
    non_alpha_ratio = sum(1 for c in stripped if not c.isalnum()) / (len(stripped) + 1e-5)

    if (
        len(text) < 30 or               # too short
        text.count(' ') < 5 or          # not enough word boundaries
        non_alpha_ratio > 0.5           # mostly symbols
    ):
        return ""

    return text

def clean_chunks_list(chunks: list[str]) -> list[str]:
    cleaned = [clean_chunk(c) for c in chunks]
    return [c for c in cleaned if c] 


def summariser(input_chunks : list) -> list:

    summaries = []

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a strict and precise summarizer for OCR-parsed documents.
                Your job is to return a concise but complete summary of the provided text, preserving all factual, logical, and numerical information. 

                Guidelines:
                - Retain all critical points, arguments, and data from the input.
                - Remove noise, irrelevant characters, and OCR-induced garbage (e.g., broken headers, misread symbols).
                - Fix broken or malformed mathematical expressions using contextual clues, and format them clearly.
                - DO NOT include any LLM-style commentary, follow-ups, soft language, or helper phrases.
                - DO NOT say anything like “This document discusses”, “In conclusion”, or “The summary is”.
                - Output only the cleaned, structured summary — as a plain string.

                Return only the final refined text. No extra output, no assistant tone."""
            ),
            (
                "user",
                "{input}"
            )
            ]
    )
    chain = prompt | llm | StrOutputParser()


    for input_chunk in input_chunks:
          
        AIMessage  = chain.invoke(
            {
                "input" : input_chunk    
            }
        )

        summaries.append(AIMessage)
    
    return(summaries)
    


