import streamlit as st
import asyncio
from main import retrieve_and_answer 

st.title('Law Buddy: Your Personal Lawyer')
st.write('Ask any Law-based question and get an answer within seconds!')

query = st.text_input('Enter your question:', key="law_query_input")

async def async_answer(query):
    return await retrieve_and_answer(query)

if st.button('Submit'):
    if query.strip():  
        result = asyncio.run(async_answer(query))  # Ensure it's awaited properly
        st.write(result)
    else:
        st.warning("Please enter a question before submitting.")
