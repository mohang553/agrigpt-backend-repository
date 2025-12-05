import asyncio
import os
from clip_service import ClipRAGService
from dotenv import load_dotenv

load_dotenv()

async def test_clip_service():
    print("Testing ClipRAGService...")
    try:
        service = ClipRAGService()
        await service.initialize()
        
        # Verify embedding dimension
        test_text = "This is a test sentence."
        embedding = service.embeddings.embed_query(test_text)
        print(f"Embedding dimension: {len(embedding)}")
        
        if len(embedding) == 512:
            print("SUCCESS: Embedding dimension is 512.")
        else:
            print(f"FAILURE: Expected 512 dimensions, got {len(embedding)}.")
            
    except Exception as e:
        print(f"Error during test: {e}")

if __name__ == "__main__":
    asyncio.run(test_clip_service())
