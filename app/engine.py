import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Initialize Global Executor
executor = ThreadPoolExecutor(max_workers=4)

class DocumentProcessor:
    @staticmethod
    def process_file(file_path: str, vector_store: Chroma):
        """Processes a PDF or TXT file and adds it to the vector store."""
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path)
            else:
                return f"Unsupported file type: {file_path}"

            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            
            # Add metadata for source citation
            filename = os.path.basename(file_path)
            for chunk in chunks:
                chunk.metadata["source"] = filename

            vector_store.add_documents(chunks)
            return f"Successfully processed {filename}"
        except Exception as e:
            return f"Error processing {file_path}: {str(e)}"

class RAGManager:
    def __init__(self):
        self.storage_path = os.getenv('STORAGE_PATH', './vectorstore')
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Determine Embeddings
        if os.getenv('OPENAI_API_KEY') and 'your_openai_key' not in os.getenv('OPENAI_API_KEY'):
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        elif os.getenv('GOOGLE_API_KEY'):
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        else:
            # Fallback to a basic one if needed, but let's assume one is available
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        self.vector_store = Chroma(
            persist_directory=self.storage_path,
            embedding_function=self.embeddings,
            collection_name="alphapulse_rag"
        )
        
        # Determine LLM
        if os.getenv('OPENAI_API_KEY') and 'your_openai_key' not in os.getenv('OPENAI_API_KEY'):
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        elif os.getenv('GOOGLE_API_KEY'):
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        elif os.getenv('GROQ_API_KEY'):
            self.llm = ChatGroq(model="llama3-70b-8192", temperature=0)
        else:
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Modern Prompt Template
        system_prompt = (
            "You are AlphaPulse AI, a highly professional RAG assistant. "
            "Answer the user's question based ONLY on the provided context. "
            "If you don't know the answer, say you don't know. "
            "Always cite your sources by mentioning the filename at the end of your response in the format \"According to [Filename]...\"."
            "\n\n"
            "{context}"
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.rag_chain = create_retrieval_chain(self.vector_store.as_retriever(search_kwargs={"k": 5}), self.question_answer_chain)

    def query(self, question: str):
        result = self.rag_chain.invoke({"input": question})
        return result["answer"]

    def ingest_async(self, file_path: str):
        """Offloads ingestion to a background thread."""
        executor.submit(DocumentProcessor.process_file, file_path, self.vector_store)

# Global Instance
rag_manager = None

def get_rag_manager():
    global rag_manager
    if rag_manager is None:
        rag_manager = RAGManager()
    return rag_manager
