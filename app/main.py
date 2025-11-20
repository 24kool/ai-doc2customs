from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.services.hts_vector_service import hts_vector_service
import uvicorn
import os

app = FastAPI(
    title="AI Doc2Customs API",
    description="Document processing and HTS code search API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(router, prefix="/api", tags=["api"])


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 Vector Store 로드"""
    print("🚀 Starting up...")
    
    # Vector Store 초기화
    try:
        if os.path.exists("data/hts_chroma"):
            hts_vector_service.load_vector_store()
            print("✅ HTS Vector Store loaded successfully")
        else:
            print("⚠️  Vector store not found. Please run: python build_vector_store.py")
    except Exception as e:
        print(f"❌ Failed to load vector store: {e}")


@app.get("/")
async def root():
    return {
        "message": "Welcome to AI Doc2Customs API",
        "docs": "/docs",
        "health": "/api/health/vector-store"
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
