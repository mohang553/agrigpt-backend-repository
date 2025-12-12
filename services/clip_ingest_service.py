import os
import io
import asyncio
from typing import List, Tuple, Dict, Any, Optional
import requests
from dotenv import load_dotenv

# PDF and Image processing
import fitz  # pymupdf
from PIL import Image

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

# CLIP Embeddings - LAZY LOADED in _ensure_clip_loaded() to speed up server startup
# from transformers import CLIPProcessor, CLIPModel  # DO NOT IMPORT HERE
# import torch  # DO NOT IMPORT HERE

# Pinecone
from pinecone import Pinecone, ServerlessSpec

# LangSmith
from langsmith import traceable

# Local Storage (replaces R2)
from services.local_storage_service import local_storage

load_dotenv()


class ClipIngestService:
    """
    CLIP-based PDF ingestion service that:
    1. Extracts text and images separately from PDFs
    2. Embeds BOTH text and images with CLIP (512 dimensions)
    3. Stores images locally in /static/images/
    4. Stores all embeddings in single Pinecone CLIP index
    5. Uses type metadata to differentiate: type="text" or type="image"
    """
    
    def __init__(self):
        # COMMENTED OUT: Google text embeddings (now using CLIP for text too)
        # self.text_embeddings = None
        # self.text_vectorstore = None
        
        self.clip_model = None
        self.clip_processor = None
        self.pinecone_client = None
        self.clip_index = None
        self.llm = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.initialized = False
        
    async def initialize(self):
        """Initialize all components (CLIP loads lazily on first use)"""
        print("Initializing CLIP Ingest Service (CLIP-only mode with HuggingFace)...")
        
        try:
            # Initialize local storage for images
            local_storage.initialize()
            
            # COMMENTED OUT: Google text embeddings
            # self.text_embeddings = GoogleGenerativeAIEmbeddings(
            #     model="models/text-embedding-004",
            #     google_api_key=os.getenv("GOOGLE_API_KEY")
            # )
            # print("Text embeddings initialized (Google text-embedding-004)")
            
            # CLIP model will be lazy-loaded on first use to speed up server startup
            self.clip_model = None  # Lazy loaded
            self.clip_processor = None  # Lazy loaded
            print("CLIP model will be loaded on first use (lazy loading with HuggingFace)")
            
            # Initialize LLM for generating answers
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.3
            )
            print("LLM initialized (gemini-2.5-flash)")
            
            # Initialize Pinecone
            self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            
            # COMMENTED OUT: Google text index setup
            # text_index_name = os.getenv("PINECONE_INDEX", "agrigpt-backend-rag-index")
            # self._ensure_index_exists(text_index_name, 768)
            # self.text_vectorstore = PineconeVectorStore(
            #     index_name=text_index_name,
            #     embedding=self.text_embeddings,
            #     namespace="citrus_crop",
            #     pinecone_api_key=os.getenv("PINECONE_API_KEY")
            # )
            # print(f"Text vector store initialized: {text_index_name} (namespace: citrus_crop)")
            
            # Setup CLIP index (512 dimensions for CLIP embeddings - BOTH text and images)
            clip_index_name = os.getenv("PINECONE_CLIP_INDEX", "agrigpt-backend-rag-clip-index")
            self._ensure_index_exists(clip_index_name, 512)
            self.clip_index = self.pinecone_client.Index(clip_index_name)
            print(f"CLIP index initialized: {clip_index_name} (stores both text and images)")
            
            self.initialized = True
            print("CLIP Ingest Service initialized successfully (CLIP-only mode)!")

                # ADD THIS: Pre-load CLIP model during initialization
            print("Pre-loading CLIP model (this may take 2-3 minutes)...")
            self._ensure_clip_loaded()
            print("✅ CLIP model loaded and ready")
    
            self.initialized = True
            
        except Exception as e:
            print(f"Error initializing CLIP Ingest Service: {str(e)}")
            raise e
    
    def _ensure_clip_loaded(self):
        """Lazy load CLIP model on first use using HuggingFace transformers"""
        if self.clip_model is None or self.clip_processor is None:
            print("Loading CLIP model from HuggingFace (first use, this may take a moment)...")
            # Import here to avoid slow module load at startup
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            # Use openai/clip-vit-base-patch32 (512 dimensions, same as sentence-transformers)
            model_name = "openai/clip-vit-base-patch32"
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.clip_model = CLIPModel.from_pretrained(model_name)
            
            # Set to eval mode and move to appropriate device
            self.clip_model.eval()
            if torch.cuda.is_available():
                self.clip_model = self.clip_model.cuda()
                print("CLIP model loaded on GPU")
            else:
                print("CLIP model loaded on CPU")
            
            print(f"CLIP model loaded ({model_name}, 512 dimensions)")
    
    def _ensure_index_exists(self, index_name: str, dimension: int):
        """Ensure Pinecone index exists, create if not"""
        existing_indexes = [idx.name for idx in self.pinecone_client.list_indexes()]
        
        if index_name not in existing_indexes:
            print(f"Creating Pinecone index: {index_name} (dimension={dimension})")
            self.pinecone_client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    region=os.getenv("PINECONE_REGION", "us-east-1")
                )
            )
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract all text from PDF"""
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    
    def extract_images_from_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract images from PDF with page information and page text context
        
        Returns:
            List of dicts with 'image_bytes', 'page_num', 'image_index', 'page_text'
        """
        images = []
        doc = fitz.open(file_path)
        
        for page_num, page in enumerate(doc):
            # Get the text from this page (to know what disease the images relate to)
            page_text = page.get_text()
            
            # Try to extract disease/topic name from first 500 chars
            page_context = page_text[:500] if page_text else ""
            
            image_list = page.get_images(full=True)
            
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                
                try:
                    # Extract image
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Convert to PNG if needed for consistency
                    if image_ext.lower() not in ["png", "jpg", "jpeg"]:
                        img = Image.open(io.BytesIO(image_bytes))
                        buffer = io.BytesIO()
                        img.save(buffer, format="PNG")
                        image_bytes = buffer.getvalue()
                        image_ext = "png"
                    
                    images.append({
                        "image_bytes": image_bytes,
                        "page_num": page_num + 1,
                        "image_index": img_index,
                        "ext": image_ext,
                        "page_text": page_context  # Include page text for context
                    })
                    
                except Exception as e:
                    print(f"Error extracting image from page {page_num + 1}: {str(e)}")
                    continue
        
        doc.close()
        print(f"Extracted {len(images)} images from PDF with page context")
        return images
    
    def embed_text(self, text: str) -> List[float]:
        """
        NEW METHOD: Embed text using CLIP model (HuggingFace)
        CLIP can embed both text and images in the same vector space
        """
        import torch
        
        # Lazy load CLIP model on first use
        self._ensure_clip_loaded()
        
        # Process text with CLIP processor
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        
        # Move to same device as model
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Get text embedding
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            # Normalize the features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Convert to list and return
        embedding = text_features.cpu().numpy()[0].tolist()
        return embedding
    
    def embed_image(self, image_bytes: bytes) -> List[float]:
        """Embed image using CLIP model (HuggingFace)"""
        import torch
        
        # Lazy load CLIP model on first use
        self._ensure_clip_loaded()
        
        # Load image from bytes
        img = Image.open(io.BytesIO(image_bytes))
        
        # CLIP expects RGB images
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Process image with CLIP processor
        inputs = self.clip_processor(images=img, return_tensors="pt")
        
        # Move to same device as model
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Get image embedding
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to list and return
        embedding = image_features.cpu().numpy()[0].tolist()
        return embedding
    
    def store_text_embedding(self, text: str, vector_id: str, metadata: Dict[str, Any]) -> None:
        """
        NEW METHOD: Store text embedding in Pinecone CLIP index
        
        Args:
            text: The text content to embed
            vector_id: Unique ID for this vector
            metadata: Metadata dict (should include type="text")
        """
        # Generate CLIP embedding for text
        clip_embedding = self.embed_text(text)
        
        # Ensure type="text" in metadata
        metadata["type"] = "text"
        metadata["content"] = text[:1000]
        
        # Store in Pinecone CLIP index
        self.clip_index.upsert(
            vectors=[{
                "id": vector_id,
                "values": clip_embedding,
                "metadata": metadata
            }]
        )
    
    def store_image_embedding(self, image_bytes: bytes, vector_id: str, metadata: Dict[str, Any]) -> None:
        """
        NEW METHOD: Store image embedding in Pinecone CLIP index
        
        Args:
            image_bytes: The image as bytes
            vector_id: Unique ID for this vector
            metadata: Metadata dict (should include type="image")
        """
        # Generate CLIP embedding for image
        clip_embedding = self.embed_image(image_bytes)
        
        # Ensure type="image" in metadata
        metadata["type"] = "image"
        
        # Store in Pinecone CLIP index
        self.clip_index.upsert(
            vectors=[{
                "id": vector_id,
                "values": clip_embedding,
                "metadata": metadata
            }]
        )
    
    @traceable(run_type="chain")
    async def process_pdf(self, file_path: str, filename: str) -> Dict[str, Any]:
        """
        Process PDF: extract text and images, embed BOTH with CLIP and store in same index
        
        Returns:
            Dict with processing results
        """
        if not self.initialized:
            raise Exception("CLIP Ingest Service not initialized")
        
        results = {
            "filename": filename,
            "text_chunks": 0,
            "images_processed": 0,
            "images_stored": 0,
            "errors": []
        }
        
        try:
            # 1. Extract and process text with CLIP
            print(f"Processing PDF: {filename}")
            text = self.extract_text_from_pdf(file_path)
            
            if text.strip():
                chunks = self.text_splitter.split_text(text)
                
                # CHANGED: Now using CLIP embeddings instead of Google embeddings
                for i, chunk in enumerate(chunks):
                    try:
                        vector_id = f"{filename}_text_{i}"
                        
                        # Store text with CLIP embedding
                        self.store_text_embedding(
                            text=chunk,
                            vector_id=vector_id,
                            metadata={
                                "source": filename,
                                "chunk": i,
                                "total_chunks": len(chunks),
                                "type": "text",
                                # NO content in metadata - retrieved via vector search
                            }
                        )
                        
                    except Exception as e:
                        error_msg = f"Error storing text chunk {i}: {str(e)}"
                        print(error_msg)
                        results["errors"].append(error_msg)
                
                results["text_chunks"] = len(chunks)
                print(f"Stored {len(chunks)} text chunks with CLIP embeddings")
            
            # 2. Extract and process images
            images = self.extract_images_from_pdf(file_path)
            results["images_processed"] = len(images)
            
            for img_data in images:
                try:
                    # Save image locally
                    img_filename = f"{filename}_p{img_data['page_num']}_i{img_data['image_index']}.{img_data['ext']}"
                    image_url = local_storage.upload_image(
                        img_data["image_bytes"],
                        img_filename,
                        content_type=f"image/{img_data['ext']}"
                    )
                    
                    # Vector ID for image
                    vector_id = f"{filename}_img_{img_data['page_num']}_{img_data['image_index']}"
                    
                    # Truncate page_text to fit Pinecone metadata limits (40KB max)
                    page_context = img_data.get("page_text", "")[:1000]
                    
                    # Store image with CLIP embedding
                    self.store_image_embedding(
                        image_bytes=img_data["image_bytes"],
                        vector_id=vector_id,
                        metadata={
                            "source": filename,
                            "page": img_data["page_num"],
                            "image_index": img_data["image_index"],
                            "image_url": image_url or "",
                            "page_text": page_context,  # Keep for images (helps with context)
                            "type": "image"
                        }
                    )
                    
                    results["images_stored"] += 1
                    print(f"Stored image: {vector_id} -> {image_url}")
                    
                except Exception as e:
                    error_msg = f"Error processing image {img_data['page_num']}-{img_data['image_index']}: {str(e)}"
                    print(error_msg)
                    results["errors"].append(error_msg)
            
            print(f"PDF processing complete: {results}")
            return results
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    @traceable(run_type="chain")
    async def query_unified(
        self, 
        query: str, 
        top_k: int = 5,
        filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        NEW UNIFIED QUERY: Search for similar content (text or images or both)
        
        Args:
            query: Text query to search for
            top_k: Number of results to return
            filter_type: Optional filter - "text", "image", or None (search both)
            
        Returns:
            List of matching results with metadata indicating type
        """
        if not self.initialized:
            raise Exception("CLIP Ingest Service not initialized")
        
        try:
            # Lazy load CLIP model on first use
            self._ensure_clip_loaded()
            
            # Embed the query text using CLIP
            query_embedding = self.embed_text(query)
            
            # Build filter if type specified
            filter_dict = None
            if filter_type:
                filter_dict = {"type": {"$eq": filter_type}}
            
            # Search Pinecone CLIP index
            results = self.clip_index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Format results - handle both text and image types
            unified_results = []
            for match in results.matches:
                result = {
                    "score": match.score,
                    "type": match.metadata.get("type", "unknown"),
                    "source": match.metadata.get("source", ""),
                }
                
                # Add type-specific fields
                if match.metadata.get("type") == "image":
                    result.update({
                        "image_url": match.metadata.get("image_url", ""),
                        "page": match.metadata.get("page", 0),
                        "image_index": match.metadata.get("image_index", 0),
                        "page_text": match.metadata.get("page_text", "")
                    })
                elif match.metadata.get("type") == "text":
                    result.update({
                        "chunk": match.metadata.get("chunk", 0),
                        "total_chunks": match.metadata.get("total_chunks", 0)
                    })
                
                unified_results.append(result)
            
            return unified_results
            
        except Exception as e:
            raise Exception(f"Error in unified query: {str(e)}")
    
    # KEEP EXISTING METHODS for backward compatibility
    @traceable(run_type="chain")
    async def query_images(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query CLIP index for relevant images based on text
        (Uses unified query with image filter)
        """
        results = await self.query_unified(query, top_k, filter_type="image")
        
        # Format to match existing API
        images = []
        for result in results:
            if result["type"] == "image":
                images.append({
                    "score": result["score"],
                    "image_url": result.get("image_url", ""),
                    "source": result["source"],
                    "page": result.get("page", 0),
                    "image_index": result.get("image_index", 0)
                })
        
        return images
    
    @traceable(run_type="chain")
    async def query_texts(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        NEW METHOD: Query CLIP index for similar text chunks
        """
        results = await self.query_unified(query, top_k, filter_type="text")
        
        # Format text results
        texts = []
        for result in results:
            if result["type"] == "text":
                texts.append({
                    "score": result["score"],
                    "source": result["source"],
                    "chunk": result.get("chunk", 0),
                    "total_chunks": result.get("total_chunks", 0)
                })
        
        return texts
    
    @traceable(run_type="chain")
    async def ask_with_image(self, image_bytes: Optional[bytes] = None, query: str = "", media_url: Optional[str] = None, top_k: int = 3) -> Dict[str, Any]:
        """
        Answer a question about a crop image using CLIP matching + text context
        NOW searches both images AND text in same CLIP index
        Supports either direct image bytes OR media_url
        """
        if not self.initialized:
            raise Exception("CLIP Ingest Service not initialized")

        # Handle media_url if image_bytes is missing
        if not image_bytes and media_url:
            print(f"📥 Downloading image from: {media_url}")
            try:
                # Use a proper user agent to avoid being blocked by some servers
                headers = {'User-Agent': 'AgriGPT-Backend/1.0'}
                response = requests.get(media_url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('Content-Type', '').lower()
                if not content_type.startswith('image/'):
                    # Try to detect magic bytes if content-type is missing or generic
                    # But for now, user requested: "see if it is an image, else return please only send images"
                    # Strictly speaking, we should rely on content-type or try to open with PIL
                    pass 
                
                # We'll verify it's an image when we try to open it with PIL below, 
                # but let's check basic headers first to fail fast
                if 'image' not in content_type and 'application/octet-stream' not in content_type:
                     # Some signed URLs might return octet-stream, so we allow that and let PIL decide
                     # But if it's text/html, clearly wrong.
                     if 'text' in content_type or 'html' in content_type:
                         raise Exception("URL does not appear to point to an image (Content-Type: " + content_type + ")")
                
                image_bytes = response.content
                
            except requests.RequestException as e:
                raise Exception(f"Failed to download image from URL: {str(e)}")
            except Exception as e:
                raise Exception(f"Error processing media URL: {str(e)}")

        if not image_bytes:
             raise Exception("Please provide either an image file or a valid mediaUrl.")

        try:
            # Wrap entire operation with timeout
            async def _process():
                # Lazy load CLIP model on first use
                self._ensure_clip_loaded()
                print("🖼️ Step 1: CLIP model loaded, processing image...")
                
                # 1. Load and embed the uploaded image with CLIP
                try:
                    img = Image.open(io.BytesIO(image_bytes))
                except Exception:
                    raise Exception("Please only send images. The provided content could not be decoded as an image.")

                print("🧠 Step 2: Generating CLIP embedding for uploaded image...")
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                image_embedding = self.embed_image(image_bytes)
                print("🔍 Step 3: Searching Pinecone for similar content...")
                
                # 2. Search CLIP index for similar content (both images and text)
                # This is the power of CLIP: image embedding can match both images AND text!
                clip_results = self.clip_index.query(
                    vector=image_embedding,
                    top_k=top_k * 2,  # Get more results to ensure we have both types
                    include_metadata=True
                )
                print(f"📊 Step 4: Found {len(clip_results.matches)} matches. Processing results...")
                
                # 3. Separate results by type
                matched_images = []
                matched_texts = []
                
                for match in clip_results.matches:
                    if match.metadata.get("type") == "image":
                        matched_images.append({
                            "source": match.metadata.get("source", ""),
                            "page": match.metadata.get("page", 0),
                            "page_text": match.metadata.get("page_text", ""),
                            "image_url": match.metadata.get("image_url", ""),
                            "score": match.score
                        })
                    elif match.metadata.get("type") == "text":
                        matched_texts.append({
                            "source": match.metadata.get("source", ""),
                            "chunk": match.metadata.get("chunk", 0),
                            "content": match.metadata.get("content", ""),
                            "score": match.score
                        })
                
                # 4. Build context from matched results
                image_contexts = []
                text_contexts = []
                image_urls = []
                
                for img_match in matched_images[:3]:
                    if img_match["page_text"]:
                        image_contexts.append(f"[Page {img_match['page']}]: {img_match['page_text'][:300]}")
                    if img_match["image_url"]:
                        image_urls.append(img_match["image_url"])
                
                # For text matches, we need to fetch the actual content
                # Since we didn't store content in metadata, we can mention the chunks found
                for txt_match in matched_texts[:3]:
                    # Fetch the actual content from metadata
                    content = txt_match.get('content', '')
                    if content:
                        text_contexts.append(f"[{txt_match['source']}, chunk {txt_match['chunk']}]: {content[:300]}")
                
                image_context = "\n".join(image_contexts) if image_contexts else "No similar images found."
                text_context = "\n".join(text_contexts) if text_contexts else "No similar text found."
                
                confidence = clip_results.matches[0].score if clip_results.matches else 0
                
                # 5. Generate answer using LLM
                prompt = f"""You are an agricultural expert helping a farmer identify crop diseases and pests.

The farmer has uploaded an image and asked: "{query}"

**SIMILAR IMAGES FOUND (with page context):**
{image_context}

**SIMILAR TEXT CHUNKS FOUND:**
{text_context}

Top match confidence: {confidence:.0%}

**YOUR TASK:**
Based on the similar images and text found in the knowledge base:

1. **Identify the disease/pest**: What does this image most likely show?
2. **Key symptoms**: List the visual symptoms to confirm this diagnosis.
3. **Treatment**: Provide specific treatment recommendations.

Be confident in your diagnosis based on the matched content."""

                response = self.llm.invoke(prompt)
                print("✅ Step 6: Answer generated successfully!")
                return {
                    "answer": response.content,
                    "matched_sources": list(set([m["source"] for m in matched_images + matched_texts])),
                    "related_images": image_urls,
                    "confidence": confidence
                }

            return await asyncio.wait_for(_process(), timeout=120.0)
    
        except asyncio.TimeoutError:
            raise Exception("Image processing timed out after 2 minutes. Please try again.")
        except Exception as e:
            raise Exception(f"Error processing image query: {str(e)}")
    
    @traceable(run_type="chain")
    async def query_images_by_image(self, image_bytes: bytes, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query CLIP index for similar images based on an uploaded image
        """
        if not self.initialized:
            raise Exception("CLIP Ingest Service not initialized")
        
        try:
            # Lazy load CLIP model on first use
            self._ensure_clip_loaded()
            
            # Load and embed the uploaded image with CLIP
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            image_embedding = self.embed_image(image_bytes)
            
            # Search Pinecone CLIP index (filter for images only)
            results = self.clip_index.query(
                vector=image_embedding,
                top_k=top_k,
                include_metadata=True,
                filter={"type": {"$eq": "image"}}
            )
            
            # Format results
            images = []
            for match in results.matches:
                images.append({
                    "score": match.score,
                    "image_url": match.metadata.get("image_url", ""),
                    "source": match.metadata.get("source", ""),
                    "page": match.metadata.get("page", 0),
                    "image_index": match.metadata.get("image_index", 0)
                })
            
            return images
            
        except Exception as e:
            raise Exception(f"Error querying images by image: {str(e)}")
    
    @traceable(run_type="chain")
    async def hybrid_query_images(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid image query: uses CLIP text embedding to find relevant images
        (Same as query_images - kept for backward compatibility)
        """
        return await self.query_images(query, top_k)


# Singleton instance
clip_ingest_service = ClipIngestService()