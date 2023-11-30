import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

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
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    # Return the vector store
    return vector_store


    

# Main function of the app
def main() -> None:
    # Loading the environment variables
    load_dotenv()

    # Setting up the page title and icon
    st.set_page_config(page_title="Revature chat bot", page_icon=":robot_face:")

    # Adding a main header to the page
    st.header("Chat with multiple PDFs :books:")

    # Adding a simple instruction text
    st.text("Ask a question about your documents:")

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

# Makes sure the app runs when the script is executed
if __name__ == '__main__':
    main()
