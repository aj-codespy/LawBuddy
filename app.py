import streamlit as st
from main import retrieve_and_answer 

st.title('Law Buddy: Your Personal Lawyer')

st.write('Ask any Law based question and get answer within seconds!')

query = st.text_input('Enter your question:', key="law_query_input")
result = retrieve_and_answer(query)

if st.button('Enter', key='generate-button'):
    st.write(result)
