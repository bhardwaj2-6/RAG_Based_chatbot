import os 
import tempfile
from typing import List

import chainlit as cl
from chainlit.types import AskFileResponse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load API key from environment variable, or set it here (better to use .env file)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize LLM - fixed model name inconsistency
model_name = "gemma2-9b-it"
llm = ChatGroq(
    temperature=0.1,
    model_name=model_name,  # Using model_name parameter
    api_key=GROQ_API_KEY
)

# Global variable for vector store
vector_store = None

# RAG system prompt
rag_system_prompt = """You are a helpful AI assistant that answers questions based on the provided PDF documents.
Follow these guidelines:
1. Answer questions based on the context provided from the PDF
2. If the answer isn't in the context, say "I don't have enough information to answer this question based on the provided PDF"
3. Keep answers concise but informative
4. If the context contains relevant charts, tables, or figures, mention their existence in your answer
5. Format your responses with markdown for readability where appropriate

Context information from the PDF:
{context}

Question: {question}

Answer the question based only on the provided context:"""


# Function to process a PDF document
def process_pdf(file: AskFileResponse) -> List[str]:
    """Process a PDF file and split it into chunks for embedding."""
    
    # Save the file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        # Use file.path to access the uploaded file
        with open(file.path, "rb") as f:
            temp_file.write(f.read())
        temp_file_path = temp_file.name
    
    # Use PyPDFLoader from Langchain to load the PDF
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    
    # Clean up the temporary file
    os.unlink(temp_file_path)
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    
    chunks = text_splitter.split_documents(documents)
    
    return chunks


# Function to setup the RAG pipeline
def setup_rag_pipeline():
    """Setup the RAG pipeline with the vector store."""
    
    global vector_store
    
    # Ensure vector store is initialized
    if vector_store is None:
        return None
    
    # Create the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 5}
    )
    
    # Create the RAG prompt
    prompt = ChatPromptTemplate.from_template(rag_system_prompt)
    
    # Setup the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


# Chainlit setup function
@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    
    # Send an initial message
    await cl.Message(
        content="Welcome to PDF-RAG! Please upload a PDF file to get started."
    ).send()
    
    # Set up file handling for PDF uploads
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to continue",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()
    
    file = files[0]
    
    # Send a processing message
    processing_msg = cl.Message(content=f"Processing `{file.name}`...")
    await processing_msg.send()
    
    # Process the PDF
    try:
        global vector_store
        
        # Get PDF chunks
        pdf_chunks = process_pdf(file)
        
        # Create persistent directory for vector store
        persist_directory = "db/faiss_index"
        os.makedirs(os.path.dirname(persist_directory), exist_ok=True)
        
        # Create vector store with document embeddings from the PDF chunks
        vector_store = FAISS.from_documents(pdf_chunks, embedding_model)
        
        # Save the index for future use (optional)
        vector_store.save_local(persist_directory)
        
        # FIXED: Correct way to update messages in Chainlit
        processing_msg.content = f"✅ `{file.name}` processed and ready for questions!"
        await processing_msg.update()
        
        # Store the file name in the user session
        cl.user_session.set("file_name", file.name)
        
    except Exception as e:
        # Send a new message instead of trying to update the existing one
        await cl.Message(content=f"❌ Error processing `{file.name}`: {str(e)}").send()
        # Don't re-raise the exception to prevent the application from crashing


# Handle chat messages
@cl.on_message
async def main(message: cl.Message):
    """Handle user messages and generate responses."""
    
    # Check if a PDF has been processed
    if vector_store is None:
        await cl.Message(
            content="Please upload a PDF file first before asking questions."
        ).send()
        return
    
    # Get the user's question
    query = message.content
    
    # Get the RAG pipeline
    rag_chain = setup_rag_pipeline()
    
    if rag_chain is None:
        await cl.Message(
            content="Error: RAG pipeline not initialized. Please try uploading your PDF again."
        ).send()
        return
    
    # Send a thinking message
    thinking_msg = cl.Message(content="Thinking...")
    await thinking_msg.send()
    
    try:
        # Get the file name from user session
        file_name = cl.user_session.get("file_name")
        
        # Process the query through the RAG chain
        response = rag_chain.invoke(query)
        
        # FIXED: Correct way to update messages in Chainlit
        thinking_msg.content = f"Based on `{file_name}`:\n\n{response}"
        await thinking_msg.update()
        
    except Exception as e:
        # FIXED: Correct way to update messages in Chainlit
        thinking_msg.content = f"❌ Error generating response: {str(e)}"
        await thinking_msg.update()


# If running directly (for testing)
if __name__ == "__main__":
    print("This application should be run with Chainlit.")
    print("Run: 'chainlit run ragapp.py' in your terminal")