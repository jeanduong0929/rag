import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

# Function to extract text from PDF documents
def get_pdf_text(pdf_docs) -> str:
    text = ""
    # Loop through each PDF document
    for pdf in pdf_docs:
        # Read the PDF
        pdf_reader = PdfReader(pdf)
        # Loop through each page in the PDF
        for page in pdf_reader.pages:
            # Extract and append text from each page
            text += page.extract_text()
    # Return the combined text from all PDFs
    return text

# Function to split text into smaller chunks
def get_text_chunks(raw_text) -> list:
    # Setting up a text splitter with specific parameters
    text_splitter = CharacterTextSplitter(
        separator="\n",        # Split text at every newline character
        chunk_size=1000,       # Each chunk will have up to 1000 characters
        chunk_overlap=200,     # Chunks will overlap by 200 characters
        length_function=len    # Function to measure the length of text
    )
    # Split the text into chunks using the defined splitter
    chunks = text_splitter.split_text(raw_text)
    # Return the list of text chunks
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(text_chunks) -> list:
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    # Create a vector store using FAISS (Fast Approximate Nearest Neighbor Search)
    # It converts the text chunks into vector representations using the embeddings
    vector_store = FAISS.from_texts(
        texts=text_chunks, embedding=embeddings
    )
    # Return the vector store
    return vector_store

# Function to create a conversation chain
def get_conversation_chain(vector_store):
    # Initialize a Chat model from OpenAI with a specific setting
    llm = ChatOpenAI(
        temperature=0  # Setting the temperature to 0 for deterministic responses
    )
    # Set up a memory buffer for the conversation history
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    # Create a conversation chain that uses the language model and vector store for retrieving information
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,  # The language model for generating responses
        retriever=vector_store.as_retriever(),  # The retriever for fetching relevant information
        memory=memory  # The memory buffer to store conversation history
    )
    # Return the conversation chain
    return conversation_chain
    
# Function to handle user input in a Streamlit app
def handle_user_input(user_question) -> None:
    # Get the response from the conversation chain for the user's question
    response = st.session_state.conversation({'question': user_question})
    # Update the chat history in the session state
    st.session_state.chat_history = response['chat_history']

    # Display the conversation in the Streamlit app
    for i, message in enumerate(st.session_state.chat_history):
        # Alternate between user and bot messages
        if i % 2 == 0:
            st.write(f"**You:** {message}")  # Display user's message
        else:
            st.write(f"**Bot:** {message}")  # Display bot's response

# Main function of the app
def main() -> None:
    # Loading the environment variables
    load_dotenv()

    # Setting up the session state for conversation
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # Setting up the session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Setting up the page title and icon
    st.set_page_config(page_title="Revature chat bot", page_icon=":robot_face:")

    # Adding a main header to the page
    st.header("Chat with Revature PDFs Bot :books:")

    # Create a text input field in the Streamlit app for user questions
    user_question = st.text_input("Ask a question about Revature:")

    # Check if the user has entered a question
    if user_question:
        # If there is a question, process it using the handle_user_input function
        handle_user_input(user_question)

    # Creating a sidebar for extra options
    with st.sidebar:
        # Adding a section title in the sidebar
        st.subheader("Your documents")

        # File upload option for PDFs, allowing multiple files
        pdf_docs = st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

        # A button in the sidebar for processing the uploaded files
        if st.button("Submit"):
            with st.spinner("Processing your documents..."):
                # Get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # Create vector store
                vector_store = get_vector_store(text_chunks)

                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_store)

# Makes sure the app runs when the script is executed
if __name__ == '__main__':
    main()
