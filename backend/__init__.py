"""
Backend package for the Qdrant Rerank RAG Assistant.

Everything under here is imported as `backend.<subpackage>.<module>` -- e.g.
`from backend.core.vector_store import VectorDBManager`. Both entry points
(app.py for Gradio/HF Spaces, backend/main.py for FastAPI) resolve imports
from the project root, so there is exactly one import convention.

This file exists so `backend` is an explicit package rather than an implicit
namespace package. Without it Python still resolves the imports, but any other
directory named `backend` on sys.path would silently merge into this one.
"""
