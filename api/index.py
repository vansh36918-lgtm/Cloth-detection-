"""
Vercel Serverless Entrypoint for AETHER LUXE (Cloth Detection & 3D Fashion Platform)
"""
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from server import app
