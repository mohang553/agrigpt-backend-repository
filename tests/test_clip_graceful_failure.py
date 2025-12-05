import asyncio
import os
from unittest.mock import MagicMock, patch
from services.clip_service import ClipRAGService

async def test_graceful_failure():
    print("Testing ClipRAGService graceful failure...")
    
    # Mock HuggingFaceEmbeddings to raise an exception
    with patch('services.clip_service.HuggingFaceEmbeddings') as mock_embeddings:
        mock_embeddings.side_effect = Exception("Simulated initialization failure")
        
        service = ClipRAGService()
        
        # Test initialization
        print("1. Testing initialize()...")
        try:
            await service.initialize()
            print("   SUCCESS: initialize() did not raise exception.")
        except Exception as e:
            print(f"   FAILURE: initialize() raised exception: {e}")
            return

        # Verify initialized state
        if service.initialized is False:
            print("   SUCCESS: service.initialized is False.")
        else:
            print(f"   FAILURE: service.initialized is {service.initialized}.")

        if "Simulated initialization failure" in service.initialization_error:
            print("   SUCCESS: service.initialization_error contains correct message.")
        else:
            print(f"   FAILURE: service.initialization_error is {service.initialization_error}.")

        # Test process_pdf
        print("\n2. Testing process_pdf()...")
        try:
            await service.process_pdf("dummy.pdf", "dummy.pdf")
            print("   FAILURE: process_pdf() did not raise exception.")
        except Exception as e:
            if "CLIP service not initialized" in str(e):
                print("   SUCCESS: process_pdf() raised correct exception.")
            else:
                print(f"   FAILURE: process_pdf() raised unexpected exception: {e}")

        # Test query
        print("\n3. Testing query()...")
        try:
            await service.query("test query")
            print("   FAILURE: query() did not raise exception.")
        except Exception as e:
            if "CLIP service not initialized" in str(e):
                print("   SUCCESS: query() raised correct exception.")
            else:
                print(f"   FAILURE: query() raised unexpected exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_graceful_failure())
