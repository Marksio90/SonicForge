"""
SonicForge Backend Server Entry Point

This file provides the entry point for uvicorn to run the FastAPI application.
"""

from app.main import app

__all__ = ["app"]
