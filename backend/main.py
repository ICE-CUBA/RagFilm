"""FastAPI Backend for Movie RAG System."""

import os
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from backend.vector_store import MovieVectorStore
from backend.llm_client import OllamaClient


# Load environment variables
load_dotenv()

# Global instances
vector_store: Optional[MovieVectorStore] = None
llm_client: Optional[OllamaClient] = None


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")


class Citation(BaseModel):
    """Citation model."""
    title: str
    year: int
    wiki_url: str


class SearchResponse(BaseModel):
    """Search response model."""
    query: str
    answer: str
    movies: List[dict]
    citations: List[Citation]


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    vector_store_loaded: bool
    total_vectors: int
    unique_movies: int
    ollama_available: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global vector_store, llm_client
    
    # Initialize on startup
    print("Initializing Movie RAG System...")
    
    # Initialize vector store
    model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    vector_store = MovieVectorStore(model_name=model_name)
    
    # Load index
    index_path = os.getenv('FAISS_INDEX_PATH', 'data/movie_index.faiss')
    metadata_path = os.getenv('METADATA_PATH', 'data/movie_index_metadata.pkl')
    
    if os.path.exists(index_path):
        vector_store.load_index(index_path, metadata_path)
        print(f"Loaded {vector_store.get_unique_movies_count()} unique movies")
    else:
        print(f"Warning: Index not found at {index_path}")
    
    # Initialize LLM client
    ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    ollama_model = os.getenv('OLLAMA_MODEL', 'llama3')
    llm_client = OllamaClient(base_url=ollama_host, model=ollama_model)
    
    print("Movie RAG System initialized")
    
    yield
    
    # Cleanup on shutdown
    print("Shutting down Movie RAG System...")


# Create FastAPI app
app = FastAPI(
    title="Movie RAG API",
    description="Movie Recommendation System using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global vector_store, llm_client
    
    vs_loaded = vector_store is not None and vector_store.index is not None
    total_vectors = vector_store.index.ntotal if vs_loaded else 0
    unique_movies = vector_store.get_unique_movies_count() if vs_loaded else 0
    ollama_ok = llm_client.health_check() if llm_client else False
    
    return HealthResponse(
        status="healthy" if vs_loaded else "degraded",
        vector_store_loaded=vs_loaded,
        total_vectors=total_vectors,
        unique_movies=unique_movies,
        ollama_available=ollama_ok
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for movies and generate recommendations.

    Args:
        request: Search request with query and top_k

    Returns:
        Search response with answer, movies, and citations
    """
    global vector_store, llm_client
    
    if vector_store is None or vector_store.index is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not loaded"
        )
    
    # Search for movies
    try:
        movies = vector_store.search(request.query, top_k=request.top_k)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )
    
    if not movies:
        return SearchResponse(
            query=request.query,
            answer="No matching movies found for your query.",
            movies=[],
            citations=[]
        )
    
    # Generate recommendations
    if llm_client:
        answer = llm_client.create_recommendation(request.query, movies)
    else:
        # Fallback if LLM not available
        answer = "LLM not available. Here are the matching movies:"
    
    # Build citations
    citations = [
        Citation(
            title=m['Title'],
            year=m['Release Year'],
            wiki_url=m.get('Wiki Page', '')
        )
        for m in movies
    ]
    
    return SearchResponse(
        query=request.query,
        answer=answer,
        movies=movies,
        citations=citations
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

