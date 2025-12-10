from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os
from services.rag_service import RAGService
from services.user_service import user_service

router = APIRouter(tags=["RAG"])

# Initialize RAG service instance (initialization happens in main.py)
rag_service = RAGService()

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    response: str
    sources: List[str]


class UserEnsureRequest(BaseModel):
    email: str


class UserEnsureResponse(BaseModel):
    userType: str

@router.post("/upload-crop-data", response_model=dict)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF file for CITRUS CROP"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process the PDF with document_type="citrus"
        num_chunks = await rag_service.process_pdf(temp_file_path, file.filename, "citrus")
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return {
            "message": f"Citrus crop PDF processed successfully. Added {num_chunks} chunks to knowledge base.",
            "filename": file.filename,
            "chunks": num_chunks
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@router.post("/ask-consultant", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the RAG system about CITRUS CROP"""
    try:
        response, sources = await rag_service.query(request.query, "citrus", request.chat_history)
        return ChatResponse(response=response, sources=sources)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@router.post("/upload-government-schemes", response_model=dict)
async def upload_government_schemes(file: UploadFile = File(...)):
    """Upload and process government schemes file"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process the PDF with document_type="schemes"
        num_chunks = await rag_service.process_pdf(temp_file_path, file.filename, "schemes")
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return {
            "message": f"Government schemes processed successfully. Added {num_chunks} chunks to knowledge base.",
            "filename": file.filename,
            "chunks": num_chunks
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@router.post("/query-government-schemes", response_model=ChatResponse)
async def query_government_schemes(request: ChatRequest):
    """Query government schemes"""
    try:
        response, sources = await rag_service.query(request.query, "schemes", request.chat_history)
        return ChatResponse(response=response, sources=sources)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@router.delete("/clear-knowledge-base")
async def clear_knowledge_base(document_type: Optional[str] = None):
    """
    Clear documents from the knowledge base
    document_type: 'citrus', 'schemes', or None (clears both)
    """
    try:
        await rag_service.clear_knowledge_base(document_type)
        if document_type:
            return {"message": f"Knowledge base cleared successfully for {document_type}"}
        else:
            return {"message": "All knowledge bases cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing knowledge base: {str(e)}")


@router.post("/users", response_model=UserEnsureResponse)
async def ensure_user(request: UserEnsureRequest):
    """
    Ensure a user exists for the given email.
    If missing, create with default userType 'user'. Returns the userType.
    """
    try:
        user_type = await user_service.ensure_user(request.email)
        return UserEnsureResponse(userType=user_type)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ensuring user: {str(e)}")