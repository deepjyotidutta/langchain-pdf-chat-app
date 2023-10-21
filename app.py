import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

def get_pdf_text(pdfs):
    # takes a pdf file and returns a list of text chunks
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    # takes a string and returns a list of text chunks
    chunks = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    ).split_text(raw_text)
    return chunks

def main():
    print("Hello World!")
    #loads the .env file
    load_dotenv()
    st.set_page_config(page_title="DevX Document Chat", page_icon="🧊", layout="wide")
    st.header("DevX Document Chat :mag_right:")
    st.text_input("Query document")
    with st.sidebar:
        st.subheader("Your documents")
        pdfs = st.file_uploader("Upload a document",accept_multiple_files=True)
        if st.button("Process document"):
            with st.spinner("Processing document"):
                # process the document
                raw_text = get_pdf_text(pdfs)
                #st.write(raw_text)
                # get text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)
                #  create vector store


if __name__ == '__main__':
    main()