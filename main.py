"""
Main Entry Point for RAG System.

Runs an example query on a document and prints the answer.
"""

from rag_agent import build_agent
import logging
import os

os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log', encoding='utf-8')
    ]
)

agent = build_agent()

print("=== RAG System ===")
print("The system will read data/document.txt, split it into chunks, and search for an answer to your question.\n")

question = input("Enter your question: ")

query = f"""
Read the file data/document.txt,
split it into chunks,
store the chunks,
and then search to answer: {question}
"""

response = agent.invoke({"input": query})

print("\n" + "="*60)
print("FINAL ANSWER:")
print("="*60)
print(response["output"])
print("="*60)
