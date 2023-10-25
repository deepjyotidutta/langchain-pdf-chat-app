import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import bot_template,user_template,css
from langchain.vectorstores import Milvus

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
    # chunks = CharacterTextSplitter(
    #     separator="\n",
    #     chunk_size=1024,
    #     chunk_overlap=200,
    #     length_function=len,
    # ).split_text(raw_text)
    chunks = CharacterTextSplitter(
        separator="\n",
        chunk_size=1024,
        chunk_overlap=200,
        length_function=len,
    ).split_text(raw_text)
    print(chunks)
    return chunks

def get_docs(raw_text):
    chunks = CharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=0
    ).split_text(raw_text)
    print(chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    docs = text_splitter.create_documents(chunks)
    return docs

def get_vectorstore(docs):
    #embeddings = OpenAIEmbeddings()
    embeddings = OpenAIEmbeddings(deployment="devx-text-embedding-ada-002-2")
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    print(embeddings)
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore

def get_vectorstore_milvus(docs):
    #embeddings = OpenAIEmbeddings()
    embeddings = OpenAIEmbeddings(deployment="devx-text-embedding-ada-002-2")
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    print(embeddings)
    vectorstore = Milvus.from_documents(
        docs,
        embedding=embeddings,
        collection_name = 'poc_collection_x',
        connection_args={"host": os.getenv("MILVUS_HOST"), "port": os.getenv("MILVUS_PORT")}
    )
    return vectorstore

def get_conversation_chain(vectorstore):
    llm=AzureChatOpenAI(
        deployment_name="gpt-4-cbre",
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    print("Hello World!")
    #loads the .env file
    load_dotenv()
    st.set_page_config(page_title="DevX Document Chat", page_icon="🧊", layout="wide")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.write(css, unsafe_allow_html=True)
    st.header("DevX Document Chat :mag_right:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)


    #st.write(user_template.replace("{{MSG}}", "Hello, Bot"),unsafe_allow_html=True)
    #st.write(bot_template.replace("{{MSG}}", "Hello, I am DevX Document Chat. I can help you find information in your documents."),unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("Your documents")
        pdfs = st.file_uploader("Upload a document",accept_multiple_files=True)
        if st.button("Process document"):
            with st.spinner("Processing document"):
                # process the document
                raw_text = get_pdf_text(pdfs)
                st.write(raw_text)
                st.write("-------------------")
                # get text chunks
                # text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                docs =  get_docs(raw_text)
                st.write("-------------------")
                st.write(docs)

                #  create vector store
                #vectorstore = get_vectorstore(docs)
                vectorstore = get_vectorstore_milvus(docs)
                st.write(vectorstore)

                #embeddings = OpenAIEmbeddings(deployment="gpt-4-cbre")
                #query_result = embeddings.embed_query(text_chunks[0])
                #st.write(query_result)
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()