import streamlit as st

# Main function of the app
def main() -> None:
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
        st.file_uploader("Upload your PDFs", type="pdf", accept_multiple_files=True)

        # A button in the sidebar for processing the uploaded files
        st.button("Submit")

# Makes sure the app runs when the script is executed
if __name__ == '__main__':
    main()
