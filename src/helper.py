import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI  # ✅ Replaced GooglePalm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY is missing! Check your .env file.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDFs."""
    if not pdf_docs:
        return None

    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            print(f"Error reading PDF {pdf.name}: {e}")
            return None

    return text if text.strip() else None


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


def get_conversational_chain(vector_store):
    """Create a conversational chain using OpenAI GPT model."""
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)  # ✅ Updated to OpenAI
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=memory
    )
    return conversation_chain





# import os
# from dotenv import load_dotenv
# from langchain.embeddings import GooglePalmEmbeddings

# # Load environment variables
# load_dotenv()

# # Print API Key (for debugging)
# print("GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))  # Ensure this prints correctly

# def get_vector_store(text_chunks):
#     # Fetch API Key
#     google_api_key = os.getenv("GOOGLE_API_KEY")

#     if not google_api_key:
#         raise ValueError("❌ GOOGLE_API_KEY is missing! Check your .env file.")

#     # Pass the API Key explicitly
#     embeddings = GooglePalmEmbeddings(google_api_key=google_api_key)
    
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     return vector_store
