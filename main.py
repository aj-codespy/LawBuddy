import faiss
import numpy as np
import asyncio
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import json
import os
import streamlit as st
from dotenv import load_dotenv


model = SentenceTransformer('all-MiniLM-L6-v2')

def load_vector_db(index_path="faissdb.idx", chunks_file="faiss_chunks.json"):
    if not os.path.exists(index_path) or not os.path.exists(chunks_file):
        raise FileNotFoundError("FAISS index or text chunks file is missing!")

    index = faiss.read_index(index_path)
    with open(chunks_file, "r") as f:
        text_chunks = json.load(f)
    return index, text_chunks

def get_text_embeddings(text):
    if not text.strip():
        return np.zeros(384)
    return model.encode([text])

async def answer_generation(input_text):
    google_api_key = 'AIzaSyDtB4bETfNDyvpzA_NnBKMrr56rdiOE8bQ'
    if not google_api_key:
        raise ValueError("Missing Google API Key! Set it in .env")

    llm = ChatGoogleGenerativeAI(
        model='gemini-1.5-flash',
        temperature=0,
        google_api_key=google_api_key,
        max_tokens=None,
        timeout=30,
        max_retries=2
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a specialist Lawyer. Your task is to answer the given question with best law precision based on the context provided for Indian constitution. If the question doesn't fit with your role or expertise, say that Sorry, I can't help with it as It's not a Law based question."),
        ("human", "{Question}")
    ])
    
    chain = prompt | llm
    return await chain.ainvoke({"Question": input_text})

async def query_vector_db_with_rag(query_text, index, text_chunks, k=3):
    if not query_text.strip():
        return "Error: Query cannot be empty!"

    query_embedding = np.array(get_text_embeddings(query_text)).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    retrieved_chunks = [text_chunks[i] for i in indices[0]]
    context = "\n".join(retrieved_chunks)

    return await answer_generation(f"Context: {context}\nQuestion: {query_text}")

async def retrieve_and_answer(query_text, index_path="faissdb.idx", chunks_file="faiss_chunks.json", k=3):
    index, stored_chunks = load_vector_db(index_path, chunks_file)
    return await query_vector_db_with_rag(query_text, index, stored_chunks, k)

# Example Usage
if __name__ == "__main__":
    query = "What are the legal aspects of employment contracts in India?"
    result = asyncio.run(retrieve_and_answer(query))
    print("Answer:", result)


