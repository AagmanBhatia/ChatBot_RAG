import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain 
from streamlit_chat import message



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function=len
        )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    converation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return converation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, mess in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            message(mess.content, is_user = True)
        else:
            message(mess.content)


def main():
    load_dotenv()
    link = "https://www.google.com/url?sa=i&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FEmblem_of_Lakshadweep&psig=AOvVaw0EHNTIsrMJjfpoDVWMEHhu&ust=1707716868357000&source=images&cd=vfe&opi=89978449&ved=0CBMQjRxqFwoTCLjDrJ3LooQDFQAAAAAdAAAAABAE"
    st.set_page_config(page_title = "Lakshadweep ChatBot", page_icon="link")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Lakshadweep ChatBot :link:")
    user_question = st.text_input("Ask a question about Lakshadweep:")
    if user_question:
        handle_userinput(user_question)


    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on process", accept_multiple_files = True)
        if st.button("Process"):
            with st.spinner("Processing"):

                # get pdf text
                raw_text = get_pdf_text(pdf_docs)


                #get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                #create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

    message("Hello! How are you?")
    message("I'm good!", is_user=True)

if __name__ == '__main__':
    main()