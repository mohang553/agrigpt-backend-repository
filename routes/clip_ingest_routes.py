from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Form
from pydantic import BaseModel
from typing import List, Optional, Union
import tempfile
import os
from services.clip_ingest_service import clip_ingest_service

router = APIRouter(tags=["CLIP Ingestion"])


class IngestResponse(BaseModel):
    message: str
    filename: str
    text_chunks: int
    images_processed: int
    images_stored: int
    errors: List[str]


class ImageQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class ImageResult(BaseModel):
    score: float
    image_url: str
    source: str
    page: int
    image_index: int


class TextResult(BaseModel):
    score: float
    source: str
    chunk: int
    total_chunks: int


class UnifiedResult(BaseModel):
    score: float
    type: str  # "text" or "image"
    source: str
    # Image-specific fields (optional)
    image_url: Optional[str] = None
    page: Optional[int] = None
    image_index: Optional[int] = None
    page_text: Optional[str] = None
    # Text-specific fields (optional)
    chunk: Optional[int] = None
    total_chunks: Optional[int] = None


class ImageQueryResponse(BaseModel):
    query: str
    results: List[ImageResult]
    count: int


class TextQueryResponse(BaseModel):
    query: str
    results: List[TextResult]
    count: int


class UnifiedQueryResponse(BaseModel):
    query: str
    results: List[UnifiedResult]
    count: int
    images_count: int
    texts_count: int


@router.post("/clip-ingest-data", response_model=IngestResponse)
async def ingest_pdf_with_clip(file: UploadFile = File(...)):
    """
    Ingest PDF with CLIP processing:
    1. Extract text and images separately
    2. Embed BOTH text and images with CLIP (512 dimensions)
    3. Store images locally
    4. Store all embeddings in single Pinecone CLIP index
    5. Use type metadata to differentiate: type="text" or type="image"
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process the PDF
        results = await clip_ingest_service.process_pdf(temp_file_path, file.filename)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return IngestResponse(
            message=f"PDF processed successfully. {results['text_chunks']} text chunks, {results['images_stored']} images stored (all with CLIP embeddings).",
            filename=results["filename"],
            text_chunks=results["text_chunks"],
            images_processed=results["images_processed"],
            images_stored=results["images_stored"],
            errors=results["errors"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@router.post("/clip-unified-query", response_model=UnifiedQueryResponse)
async def unified_query(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(10, description="Number of results to return"),
    filter_type: Optional[str] = Query(None, description="Filter by type: 'text', 'image', or None for both")
):
    """
    NEW UNIFIED QUERY: Search for similar content (text, images, or both)
    
    Uses CLIP embeddings to find semantically similar content regardless of type.
    This enables powerful cross-modal search:
    - Text query → finds both similar text AND related images
    - Can filter to only text or only images if needed
    
    Examples:
    - "citrus leaf disease" → returns disease text descriptions + disease images
    - "treatment methods" → returns treatment text + treatment diagrams
    """
    if filter_type and filter_type not in ["text", "image"]:
        raise HTTPException(status_code=400, detail="filter_type must be 'text', 'image', or None")
    
    try:
        results = await clip_ingest_service.query_unified(query, top_k, filter_type)
        
        # Count by type
        images_count = sum(1 for r in results if r["type"] == "image")
        texts_count = sum(1 for r in results if r["type"] == "text")
        
        return UnifiedQueryResponse(
            query=query,
            results=[UnifiedResult(**r) for r in results],
            count=len(results),
            images_count=images_count,
            texts_count=texts_count
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in unified query: {str(e)}")


@router.post("/clip-query-texts", response_model=TextQueryResponse)
async def query_texts(request: ImageQueryRequest):
    """
    Query CLIP index for similar TEXT chunks based on text query
    
    Uses CLIP text embeddings for semantic text-to-text search.
    """
    try:
        texts = await clip_ingest_service.query_texts(request.query, request.top_k)
        
        return TextQueryResponse(
            query=request.query,
            results=[TextResult(**txt) for txt in texts],
            count=len(texts)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying texts: {str(e)}")


# EXISTING ENDPOINTS (kept for backward compatibility)

@router.post("/clip-query-data", response_model=ImageQueryResponse)
async def query_clip_images(request: ImageQueryRequest):
    """
    Query CLIP index for relevant images based on text
    (Backward compatible - uses unified query with image filter)
    """
    try:
        images = await clip_ingest_service.query_images(request.query, request.top_k)
        
        return ImageQueryResponse(
            query=request.query,
            results=[ImageResult(**img) for img in images],
            count=len(images)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying images: {str(e)}")


class AskWithImageResponse(BaseModel):
    answer: str
    matched_sources: List[str]
    related_images: List[str]
    confidence: float


@router.post("/ask-with-image", response_model=AskWithImageResponse)
async def ask_with_image(
    file: Optional[UploadFile] = File(None),
    mediaUrl: Optional[str] = Form(None),
    query: str = Form("What disease does this crop have and how can I treat it?")
):
    """
    Answer questions about a crop image:
    1. Upload a crop image OR provide mediaUrl
    2. CLIP finds similar content (BOTH images AND text) in database
    3. LLM generates expert answer using matched content
    
    NOW ENHANCED: Searches both images and text in same CLIP index!
    """
    # Validate that either file or mediaUrl is provided
    if not file and not mediaUrl:
        raise HTTPException(status_code=400, detail="Please provide either an image file or a mediaUrl")

    # Validate file type if present
    if file:
        allowed_extensions = [".jpg", ".jpeg", ".png", ".webp"]
        file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Only image files (JPEG, PNG, WebP) are allowed")
    
    try:
        image_bytes = None
        if file:
            image_bytes = await file.read()
        
        # Service handles media_url download if image_bytes is None
        result = await clip_ingest_service.ask_with_image(image_bytes=image_bytes, query=query, media_url=mediaUrl)
        
        return AskWithImageResponse(
            answer=result["answer"],
            matched_sources=result["matched_sources"],
            related_images=result["related_images"],
            confidence=result["confidence"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image query: {str(e)}")


@router.post("/hybrid-image-query", response_model=ImageQueryResponse)
async def hybrid_image_query(request: ImageQueryRequest):
    """
    Hybrid text-to-image query
    (Same as clip-query-data - kept for backward compatibility)
    """
    try:
        images = await clip_ingest_service.hybrid_query_images(request.query, request.top_k)
        
        return ImageQueryResponse(
            query=request.query,
            results=[ImageResult(**img) for img in images],
            count=len(images)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in hybrid image query: {str(e)}")


class ImageSearchResponse(BaseModel):
    results: List[ImageResult]
    count: int


@router.post("/query-by-image", response_model=ImageSearchResponse)
async def query_by_image(
    file: UploadFile = File(...),
    top_k: int = 5
):
    """
    Image-to-image search:
    1. Upload an image
    2. Find similar images in the database using CLIP
    3. Return matching images with URLs
    """
    # Validate file type
    allowed_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Only image files (JPEG, PNG, WebP) are allowed")
    
    try:
        image_bytes = await file.read()
        
        images = await clip_ingest_service.query_images_by_image(image_bytes, top_k)
        
        return ImageSearchResponse(
            results=[ImageResult(**img) for img in images],
            count=len(images)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in image search: {str(e)}")