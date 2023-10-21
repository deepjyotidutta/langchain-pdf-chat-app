import streamlit as st
from dotenv import load_dotenv


def main():
    print("Hello World!")
    #loads the .env file
    load_dotenv()
    st.set_page_config(page_title="DevX Document Chat", page_icon="🧊", layout="wide")
    st.header("DevX Document Chat")
    st.text_input("Query document")
    with st.sidebar:
        st.subheader("Your documents")
        st.file_uploader("Upload a document")
        st.button("Process document")

if __name__ == '__main__':
    main()