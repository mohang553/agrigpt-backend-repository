import os
from typing import List, Tuple, Optional, Any, Dict
from dotenv import load_dotenv

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# LangSmith imports
from langsmith import traceable

# PDF processing
from pypdf import PdfReader

# Pinecone
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

class ClipRAGService:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.initialized = False
        self.initialization_error = None
        
    async def initialize(self):
        """Initialize all components for CLIP RAG"""
        print("Initializing CLIP RAG Service...")
        
        try:
            # Initialize CLIP embeddings
            # Using sentence-transformers/clip-ViT-B-32 which has 512 dimensions
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/clip-ViT-B-32"
            )
            
            if self.embeddings is None:
                raise ValueError("CLIP Embeddings initialization failed")
            else:
                print("CLIP Embeddings initialized successfully")
            
            # Initialize LLM (still using Gemini for the chat part)
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.3,
                convert_system_message_to_human=True
            )
            
            # Initialize Pinecone
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            
            index_name = os.getenv("PINECONE_CLIP_INDEX", "agrigpt-backend-rag-index-clip")
            
            # Check if index exists, create if not
            existing_indexes = [index.name for index in pc.list_indexes()]
            
            if index_name not in existing_indexes:
                print(f"Creating new Pinecone index: {index_name}")
                pc.create_index(
                    name=index_name,
                    dimension=512,  # Dimension for CLIP ViT-B-32
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
            
            # Initialize vector store
            self.vectorstore = PineconeVectorStore(
                index_name=index_name,
                embedding=self.embeddings,
                pinecone_api_key=os.getenv("PINECONE_API_KEY")
            )
            
            if self.vectorstore is None:
                raise ValueError("Vector store initialization failed")
            else:
                print("Vector store initialized successfully")
                
            self.initialized = True
            
        except Exception as e:
            print(f"CLIP RAG Service initialization failed: {str(e)}")
            self.initialization_error = str(e)
            self.initialized = False
    
    async def process_pdf(self, file_path: str, filename: str) -> int:
        """Process PDF and add to CLIP vector store"""
        if not self.initialized:
            raise Exception(f"CLIP service not initialized: {self.initialization_error}")
            
        try:
            # Read PDF
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "chunk": i,
                        "total_chunks": len(chunks),
                        "embedding_model": "clip-ViT-B-32"
                    }
                )
                documents.append(doc)
            
            # Add documents to vector store
            self.vectorstore.add_documents(documents)
            
            return len(documents)
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    @traceable(run_type="retriever")
    def retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve documents relevant to the query using CLIP embeddings"""
        print("begin retrieve_documents (CLIP)")
        import time
        retries = 3
        docs = []
        for attempt in range(retries):
            try:
                docs = self.vectorstore.similarity_search(query, k=5)
                break
            except Exception as e:
                if "429" in str(e) and attempt < retries - 1:
                    print(f"Rate limit hit, retrying in {2 * (attempt + 1)} seconds...")
                    time.sleep(2 * (attempt + 1))
                    continue
                raise e
        print("end retrieve_documents (CLIP)")
        return [
            {
                "page_content": doc.page_content,
                "type": "Document",
                "metadata": doc.metadata
            }
            for doc in docs
        ]

    @traceable(run_type="prompt")
    def create_prompt(self, query: str, context: List[Dict[str, Any]], chat_history: List[dict] = None) -> List[Any]:
        """Create the prompt for the LLM"""
        # Reuse the same prompt logic as the main service
        context_str = "\n\n".join([doc["page_content"] for doc in context])
        
        system_prompt = (
            "You are a helpful assistant. Use the following pieces of context to answer the user's question. "
            "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
            f"Context:\n{context_str}"
        )
        
        messages = [("system", system_prompt)]
        
        if chat_history:
            for item in chat_history:
                if item.get("role") == "user":
                    messages.append(("human", item.get("content", "")))
                elif item.get("role") == "assistant":
                    messages.append(("ai", item.get("content", "")))
        
        messages.append(("human", query))
        
        prompt_template = ChatPromptTemplate.from_messages(messages)
        return prompt_template.format_messages()

    @traceable(run_type="llm")
    def call_llm(self, messages: List[Any]) -> str:
        """Call the LLM with the messages"""
        print("begin call_llm")
        response = self.llm.invoke(messages)
        print("end call_llm")
        return response.content

    @traceable(run_type="chain")
    async def query(self, query: str, chat_history: List[dict] = None) -> Tuple[str, List[str]]:
        """Query the CLIP RAG system"""
        if not self.initialized:
            raise Exception(f"CLIP service not initialized: {self.initialization_error}")
            
        try:
            print("Query (CLIP): ", query)
            # 1. Retrieve documents
            retrieved_docs = self.retrieve_documents(query)
            print("Retrieved documents: ", retrieved_docs)
            # 2. Create prompt
            messages = self.create_prompt(query, retrieved_docs, chat_history)
            print("Messages: ", messages)
            # 3. Call LLM
            answer = self.call_llm(messages)
            print("Answer: ", answer)
            # Extract sources
            sources = []
            for doc in retrieved_docs:
                metadata = doc.get("metadata", {})
                source = metadata.get("source", "Unknown")
                chunk = metadata.get("chunk", 0)
                sources.append(f"{source} (chunk {chunk + 1})")
            
            return answer, sources
            
        except Exception as e:
            raise Exception(f"Error querying CLIP RAG system: {str(e)}")
