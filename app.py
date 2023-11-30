import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

def get_pdf_text(pdf_docs) -> str:
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text) -> list:
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks
    

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

                st.write(raw_text)

                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # Create vector store

# Makes sure the app runs when the script is executed
if __name__ == '__main__':
    main()
