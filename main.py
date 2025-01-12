import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import json
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()



model = SentenceTransformer('all-MiniLM-L6-v2')

def load_vector_db(index_path="faissdb.idx", chunks_file="faiss_chunks.json"):
    index = faiss.read_index(index_path)
    with open(chunks_file, "r") as f:
        text_chunks = json.load(f)
    return index, text_chunks

def get_text_embeddings(text):
    if not text.strip():
        return np.zeros(384)
    embeddings = model.encode([text])
    return embeddings

def answer_generation(input):
    llm = ChatGoogleGenerativeAI(
        model='gemini-1.5-flash',
        temperature=0,
        api_key=os.getenv("API_KEY"),
        max_tokens=None,
        timeout=30,
        max_retries=2
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a specialist Lawyer. Your task is to answer the given question with best law precison based on the context provided for Indian constitution. If the question doesn't fit with your role or expertise, say that Sorry, I can't help with it as It's not a Law based question and don't answer anything. You've to answer all kinds of law related questions like employment contracts, confidentiality agreements etc, be as useful as you can be for the customer"),
        ("human", "{Question}")
    ])
    chain = prompt | llm
    result = chain.invoke({"Question": input})
    return result.content

def query_vector_db_with_rag(query_text, index, text_chunks, k=3):
    query_embedding = np.array(get_text_embeddings(query_text)).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    retrieved_chunks = [text_chunks[i] for i in indices[0]]
    context = "\n".join(retrieved_chunks)
    response = answer_generation(f"Context: {context}\nQuestion: {query_text}")
    return response

def retrieve_and_answer(query_text, index_path="faissdb.idx", chunks_file="faiss_chunks.json", k=3):
    index, stored_chunks = load_vector_db(index_path, chunks_file)
    result = query_vector_db_with_rag(query_text, index, stored_chunks, k)
    return result

# query = "I'm hungry and fat, what should I eat?"
# result = retrieve_and_answer(query)
# print("Answer:", result)

st.title('Law Buddy: Your Personal Lawyer')

st.write('Ask any Law based question and get answer within seconds!')

question = st.text_input('Enter your question: ')

if st.button('Enter'):
    st.write(retrieve_and_answer(question))