import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# FAISS Import
from langchain_community.vectorstores import FAISS
# Conversational Chain Import
from langchain.chains import ConversationalRetrievalChain

from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing in environment variables.")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize Google Gemini model
llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.7)  # Gemini-Pro Model

# Initialize Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Latest Embedding Model


def get_pdf_text(pdf_docs):
    """Extracts text from a list of PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle NoneType cases
    return text


def get_text_chunks(text):
    """Splits the extracted text into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    """Creates a FAISS vector store from text chunks using Google Gemini embeddings."""
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


def get_conversational_chain(vector_store):
    """Creates a Conversational Retrieval Chain for interactive Q&A."""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
    )
    return conversation_chain
