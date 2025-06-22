from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
import dotenv

dotenv.load_dotenv("ai_layer/keys.env")
GROQ_API_KEY = os.getenv("GROQ_SUMM")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


def complete_summary(summaries: list) -> str:
    text = ""
    for summary in summaries:
        text += " "
        text += summary
    
    return text

quiz_prompt = """
                You are a precise quiz generator.

                Given a summary of an OCR-parsed document, generate a factual and unambiguous quiz to test understanding of its contents.

                Guidelines:
                - Ask only questions based on information explicitly present in the input.
                - Focus on factual recall, numerical values, definitions, key concepts, and logic derived from the text.
                - Avoid vague or opinion-based questions.
                - Use only these formats: "multiple_choice", "short_answer", or "true_false".
                - Do not include helper phrases, soft language, or assistant commentary.

                Output Format:
                Return a valid JSON array of question objects, where each object includes:
                - "question": string
                - "type": one of ["multiple_choice", "short_answer", "true_false"]
                - "options": array (empty for short_answer and true_false)
                - "answer": correct answer(s), as a string or list of strings

                Example Output:
                If the summary is:  
                "Chemotherapy drugs target rapidly dividing cells. Side effects include nausea, fatigue, and hair loss. Dosage varies based on weight and cancer type."

                The output must be:
                [
                {
                    "question": "What type of cells are primarily targeted by chemotherapy drugs?",
                    "type": "multiple_choice",
                    "options": [
                    "Slow-dividing cells",
                    "Rapid-dividing cells",
                    "Blood cells",
                    "Skin cells"
                    ],
                    "answer": "Rapid-dividing cells"
                },
                {
                    "question": "List two common side effects of chemotherapy.",
                    "type": "short_answer",
                    "options": [],
                    "answer": ["Nausea", "Fatigue", "Hair loss"]
                },
                {
                    "question": "True or False: Chemotherapy dosage is the same for all patients.",
                    "type": "true_false",
                    "options": [],
                    "answer": "False"
                }
                ]

                Return only the JSON. No explanations, no preface, no follow-up text."""                """
                You are a precise quiz generator.

                Given a summary of an OCR-parsed document, generate a factual and unambiguous quiz to test understanding of its contents.

                Guidelines:
                - Ask only questions based on information explicitly present in the input.
                - Focus on factual recall, numerical values, definitions, key concepts, and logic derived from the text.
                - Avoid vague or opinion-based questions.
                - Use only these formats: "multiple_choice", "short_answer", or "true_false".
                - Do not include helper phrases, soft language, or assistant commentary.

                Output Format:
                Return a valid JSON array of question objects, where each object includes:
                - "question": string
                - "type": one of ["multiple_choice", "short_answer", "true_false"]
                - "options": array (empty for short_answer and true_false)
                - "answer": correct answer(s), as a string or list of strings

                Example Output:
                If the summary is:  
                "Chemotherapy drugs target rapidly dividing cells. Side effects include nausea, fatigue, and hair loss. Dosage varies based on weight and cancer type."

                The output must be:
                [
                {
                    "question": "What type of cells are primarily targeted by chemotherapy drugs?",
                    "type": "multiple_choice",
                    "options": [
                    "Slow-dividing cells",
                    "Rapid-dividing cells",
                    "Blood cells",
                    "Skin cells"
                    ],
                    "answer": "Rapid-dividing cells"
                },
                {
                    "question": "List two common side effects of chemotherapy.",
                    "type": "short_answer",
                    "options": [],
                    "answer": ["Nausea", "Fatigue", "Hair loss"]
                },
                {
                    "question": "True or False: Chemotherapy dosage is the same for all patients.",
                    "type": "true_false",
                    "options": [],
                    "answer": "False"
                }
                ]

                Return only the JSON. No explanations, no preface, no follow-up text."""

flashcard_prompt = """
You are a precise flashcard generator.

Given a summary of an OCR-parsed document, extract key factual concepts and convert them into concise flashcards.

Guidelines:
- Each flashcard must contain one clear question and a direct answer.
- Focus on facts, definitions, formulas, data, reasoning steps, and terminology from the input.
- Avoid opinion-based or vague content.
- Do not generate assistant-style filler, instructions, or commentary.

Output Format:
Return a valid JSON array of flashcards. Each flashcard must include:
- "question": a concise, standalone question
- "answer": a clear and factual answer

Example Output:
If the summary is:
"Chemotherapy drugs target rapidly dividing cells. Side effects include nausea, fatigue, and hair loss. Dosage varies based on weight and cancer type."

The output must be:
[
  {
    "question": "What type of cells do chemotherapy drugs target?",
    "answer": "Rapidly dividing cells"
  },
  {
    "question": "What are three common side effects of chemotherapy?",
    "answer": "Nausea, fatigue, and hair loss"
  },
  {
    "question": "What factors influence chemotherapy dosage?",
    "answer": "Patient weight and cancer type"
  }
]

Return only the JSON. No additional explanations, labels, or assistant output.
"""

def generate_quiz(summary : str):

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "{sys_prompt}"
            ),
            (
                "user",
                "{input}"
            )
            ]
    )
    chain = prompt | llm | StrOutputParser()

          
    AIMessage  = chain.invoke(
        {
            "sys_prompt": quiz_prompt,
            "input" : summary    
        }
    )

    return(AIMessage)

def generate_flash(summary : str):

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "{sys_prompt}"
            ),
            (
                "user",
                "{input}"
            )
            ]
    )
    chain = prompt | llm | StrOutputParser()

          
    AIMessage  = chain.invoke(
        {
            "sys_prompt": flashcard_prompt,
            "input" : summary    
        }
    )

    return(AIMessage)

