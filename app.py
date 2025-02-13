import sys
import os
import streamlit as st
import time

# Add the absolute path to src
sys.path.append(r"C:\BVDATA\OneDrive - Bureau Veritas\GSK_data\Soft_BA\Information-Retrivel-Sytem\src")

from helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain

# Set Streamlit page configuration
st.set_page_config(page_title="Information Retrieval", layout="wide")


def user_input(user_question):
    """Handles user queries and retrieves responses from the conversational model."""
    if st.session_state.conversation:
        response = st.session_state.conversation({"query": user_question})
        st.session_state.chatHistory = response.get("chat_history", [])

        # Display conversation history
        for i, message in enumerate(st.session_state.chatHistory):
            if i % 2 == 0:
                st.write("👤 **User**: ", message.content)
            else:
                st.write("🤖 **Reply**: ", message.content)
    else:
        st.error("❌ Please upload and process PDFs first!")


def main():
    st.header("📄 Information Retrieval System 🚀")
    st.write("Upload PDF files and ask questions to retrieve relevant information!")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = []

    # User question input
    user_question = st.text_input("🔍 Ask a question from the PDF Files:")

    if user_question:
        user_input(user_question)

    # Sidebar for file upload
    with st.sidebar:
        st.title("📂 Menu:")
        pdf_docs = st.file_uploader("Upload PDF files and click Submit & Process", accept_multiple_files=True, type=["pdf"])

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("🔄 Processing..."):
                    time.sleep(2)

                    # Extract text and process
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        vector_store = get_vector_store(text_chunks)
                        st.session_state.conversation = get_conversational_chain(vector_store)
                        st.success("✅ Processing Complete!")
                    else:
                        st.error("❌ No extractable text found in PDFs!")
            else:
                st.error("❌ Please upload at least one PDF file!")

        # Clear chat history button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chatHistory = []
            st.success("✅ Chat history cleared!")


if __name__ == "__main__":
    main()


# import os
# from dotenv import load_dotenv

# # Load .env variables
# load_dotenv()

# # Set API key manually
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# # Import helper functions after setting API key
# from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain
