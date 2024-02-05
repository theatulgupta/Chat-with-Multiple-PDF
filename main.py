import os
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores.faiss import FAISS 
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

load_dotenv()

# Configure genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to read all PDF files and append them to text
def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        text += " ".join([page.extract_text() for page in pdf_reader.pages])
    return text

# Function to split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        length_function=len
    )
    return text_splitter.split_text(text)

# Function to get the vector store from the chunks
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to get the conversational chain   
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    Make sure to provide all the details. If the answer is not in the context, say "answer is not available in the context", 
    don't provide the wrong answers.\n
    Context: {context}\n
    Question: {question}\n
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to get the user input and generate a response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain({
        "input_documents": docs,
        "question": user_question
    }, return_only_outputs=True)

    st.write(response["output_text"])

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Chat with Multiple PDF", 
        page_icon=":robot:",
        layout="wide")
    
    st.header("Upload PDF files and ask questions about them")
    
    user_question = st.text_input("Ask a question about the PDF files")
    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu:")
        pdf_files = st.file_uploader("Upload PDF files and Click on the Submit & Process", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                text = get_pdf_text(pdf_files)
                chunks = get_text_chunks(text)
                get_vector_store(chunks)
                st.success("Processing complete!")
                
if __name__ == "__main__":
    main()
