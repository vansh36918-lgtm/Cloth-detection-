"""
AETHER LUXE — Main Application Entrypoint
Delegates to the unified high-performance FastAPI server in server.py
"""

import sys
import os

# Add root directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from server import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"==================================================")
    print(f"  AETHER LUXE — 3D Cyber-Couture & AR Try-On      ")
    print(f"  Server running live at: http://localhost:{port} ")
    print(f"==================================================")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)