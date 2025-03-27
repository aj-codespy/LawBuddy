import streamlit as st
from main import retrieve_and_answer 

st.title('Law Buddy: Your Personal Lawyer')
st.write('Ask any Law-based question and get an answer within seconds!')

query = st.text_input('Enter your question:', key="law_query_input")

if st.button('Submit'):
    if query.strip():  # Ensures the query isn't empty
        result = retrieve_and_answer(query)
        st.write(result)
    else:
        st.warning("Please enter a question before submitting.")
