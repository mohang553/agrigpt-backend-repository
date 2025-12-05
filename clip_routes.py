from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os
from clip_service import ClipRAGService

router = APIRouter(prefix="/clip", tags=["CLIP Embeddings"])

# Initialize CLIP RAG service
clip_rag_service = ClipRAGService()

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    response: str
    sources: List[str]

@router.post("/upload", response_model=dict)
async def upload_pdf_clip(file: UploadFile = File(...)):
    """Upload and process PDF file using CLIP embeddings"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process the PDF
        num_chunks = await clip_rag_service.process_pdf(temp_file_path, file.filename)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return {
            "message": f"PDF processed successfully with CLIP. Added {num_chunks} chunks to knowledge base.",
            "filename": file.filename,
            "chunks": num_chunks,
            "model": "clip-ViT-B-32"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@router.post("/ask", response_model=ChatResponse)
async def chat_clip(request: ChatRequest):
    """Chat with the CLIP RAG system"""
    try:
        response, sources = await clip_rag_service.query(request.query, request.chat_history)
        return ChatResponse(response=response, sources=sources)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
