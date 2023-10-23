import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import bot_template,user_template,css

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
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
    ).split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = OpenAIEmbeddings(deployment="devx-text-embedding-ada-002-2")
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    print(embeddings)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
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
                #st.write(raw_text)
                # get text chunks
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)
                #  create vector store
                vectorstore = get_vectorstore(text_chunks)
                #embeddings = OpenAIEmbeddings(deployment="gpt-4-cbre")
                #query_result = embeddings.embed_query(text_chunks[0])
                #st.write(query_result)
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()