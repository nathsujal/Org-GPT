"""
Hugging Face Spaces entry point for OrgGPT.

This file serves as the main entry point for running OrgGPT
on Hugging Face Spaces platform.
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# Import the FastAPI app
from backend.main import app

if __name__ == "__main__":
    # Get port from environment (Hugging Face Spaces uses 7860)
    port = int(os.getenv("PORT", 7860))
    
    # Run the application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

