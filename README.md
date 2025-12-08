# Movie RAG System

A Movie Retrieval-Augmented Generation (RAG) system that uses FAISS vector search and Llama 3 to provide movie recommendations based on natural language queries.

## Features

- **FAISS Vector Search**: Dense retrieval using all-MiniLM-L6-v2 embeddings (384 dimensions)
- **LLM Recommendations**: Ollama + Llama 3 for generating coherent recommendations
- **Wikipedia Citations**: Clickable links to movie Wikipedia pages
- **Docker Deployment**: Easy deployment with FastAPI backend and Streamlit frontend

## Dataset

This system uses the [Wikipedia Movie Plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) dataset containing 34,886 movies with:
- Title
- Genre
- Director
- Release Year
- Plot
- Wiki Page URL

## Project Structure

```
movie-rag-minimal/
├── .env
├── .gitignore
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
├── README.md
│
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI with /health and /search endpoints
│   ├── vector_store.py      # FAISS + MiniLM
│   └── llm_client.py        # Ollama client
│
├── frontend/
│   └── app.py               # Streamlit UI
│
├── scripts/
│   ├── clean_data.py        # Clean CSV
│   └── build_index.py       # Build FAISS index
│
└── data/
    ├── movies_cleaned.csv
    ├── movie_index.faiss
    └── movie_index_metadata.pkl
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- At least 6GB RAM
- (Optional) NVIDIA GPU for faster LLM inference

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd movie-rag
   ```

2. **Download the dataset**:
   
   Download `wiki_movie_plots_deduped.csv` from [Kaggle](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) and place it in the `data/` folder.

3. **Prepare the data**:
   ```bash
   # Install dependencies locally (or use Docker)
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   
   # Clean the data
   python scripts/clean_data.py --input data/wiki_movie_plots_deduped.csv --output data/movies_cleaned.csv
   
   # Build the FAISS index
   python scripts/build_index.py --input data/movies_cleaned.csv --output data/movie_index
   ```

4. **Start with Docker**:
   ```bash
   docker-compose up --build
   ```

5. **Pull the Llama 3 model** (first time only):
   ```bash
   docker exec -it movie-rag-ollama ollama pull llama3
   ```

6. **Access the application**:
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## Local Development (without Docker)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Install and run Ollama**:
   ```bash
   # Install Ollama from https://ollama.ai
   ollama serve
   ollama pull llama3
   ```

3. **Prepare data** (if not done):
   ```bash
   python scripts/clean_data.py
   python scripts/build_index.py
   ```

4. **Start the backend**:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

5. **Start the frontend** (in another terminal):
   ```bash
   streamlit run frontend/app.py
   ```

## API Endpoints

### Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "vector_store_loaded": true,
  "total_vectors": 150000,
  "unique_movies": 34000,
  "ollama_available": true
}
```

### Search Movies
```bash
POST /search
Content-Type: application/json

{
  "query": "sci-fi movie about time travel",
  "top_k": 5
}
```

Response:
```json
{
  "query": "sci-fi movie about time travel",
  "answer": "Based on your query, here are my recommendations...",
  "movies": [
    {
      "Title": "Interstellar",
      "Genre": "Science Fiction",
      "Director": "Christopher Nolan",
      "Release Year": 2014,
      "Plot": "...",
      "Wiki Page": "https://en.wikipedia.org/wiki/Interstellar_(film)",
      "similarity_score": 0.85,
      "matched_chunk": "..."
    }
  ],
  "citations": [
    {
      "title": "Interstellar",
      "year": 2014,
      "wiki_url": "https://en.wikipedia.org/wiki/Interstellar_(film)"
    }
  ]
}
```

## Testing

### Health Check
```bash
curl http://localhost:8000/health
```

### Search
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "action movie with explosions", "top_k": 3}'
```

## Performance

- **Precision@5**: ~78-80%
- **Latency**: ~2-3 seconds per query
- **Memory**: ~4GB RAM
- **Index size**: ~130MB
- **Models**: 80MB (MiniLM) + 4.7GB (Llama 3)

## Architecture

```
User Query
    │
    ▼
┌─────────────┐
│  Streamlit  │
│  Frontend   │
└─────┬───────┘
      │
      ▼
┌─────────────┐     ┌─────────────┐
│   FastAPI   │────►│   FAISS     │
│   Backend   │     │   Index     │
└─────┬───────┘     └─────────────┘
      │
      ▼
┌─────────────┐
│   Ollama    │
│   Llama 3   │
└─────────────┘
      │
      ▼
Recommendations with Wikipedia Citations
```

## Troubleshooting

### Common Issues

1. **Backend not connecting to Ollama**:
   - Make sure Ollama is running: `docker logs movie-rag-ollama`
   - Pull the model: `docker exec movie-rag-ollama ollama pull llama3`

2. **Index not found**:
   - Run the build script: `python scripts/build_index.py`
   - Check the `data/` folder for `.faiss` and `.pkl` files

3. **Out of memory**:
   - Reduce batch size in `build_index.py`
   - Use a smaller model variant

4. **Slow responses**:
   - First query loads the model (subsequent queries are faster)
   - Consider using GPU for Ollama

## License

MIT License

