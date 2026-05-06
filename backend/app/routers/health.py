"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["Health"])


@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "qai-tester-v2", "version": "0.1.0"}
