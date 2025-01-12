import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from nltk.tokenize import sent_tokenize

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def get_text_embeddings(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text)
    return embeddings

def create_and_save_faiss_index(pdf_path, output_file="faiss_index.idx"):
    pdf_text = extract_text_from_pdf(pdf_path)
    text_chunks = sent_tokenize(pdf_text)
    text_embeddings = [get_text_embeddings(chunk) for chunk in text_chunks]
    embeddings_np = np.vstack(text_embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    faiss.write_index(index, output_file)
    print('successful')

pdf_path = 'constitution.pdf'
create_and_save_faiss_index(pdf_path, "faiss_index.idx")
