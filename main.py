from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables
load_dotenv()

# Import Routers and Services
from routes.rag_routes import router as rag_router, rag_service
from routes.clip_ingest_routes import router as clip_ingest_router
from services.clip_ingest_service import clip_ingest_service

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# Readiness flag - services are not ready until initialization completes
services_ready = False
initialization_error = None

# Mount static files for serving images
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include Routers
app.include_router(rag_router)
app.include_router(clip_ingest_router)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def initialize_services_background():
    """Initialize services in background after server starts"""
    global services_ready, initialization_error
    try:
        print("🚀 Starting background initialization...")
        await rag_service.initialize()
        await clip_ingest_service.initialize()
        services_ready = True
        print("✅✅✅ ALL SERVICES READY ✅✅✅")
    except Exception as e:
        initialization_error = str(e)
        print(f"❌ Background initialization failed: {e}")


@app.on_event("startup")
async def startup_event():
    """Start background initialization - doesn't block server startup"""
    print("🌐 Server starting - port will open immediately")
    print("📦 Services will initialize in background...")
    asyncio.create_task(initialize_services_background())


@app.get("/health")
async def health_check():
    """Health check endpoint - always responds (for Render port detection)"""
    return {
        "status": "healthy",
        "service": "RAG Chatbot API",
        "services_ready": services_ready
    }


@app.get("/ready")
async def readiness_check():
    """Readiness check - returns 503 if services are still initializing"""
    if initialization_error:
        raise HTTPException(status_code=503, detail=f"Initialization failed: {initialization_error}")
    if not services_ready:
        raise HTTPException(status_code=503, detail="Services are still initializing...")
    return {"status": "ready", "services_ready": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

