# AI Doc2Customs - Document Processing & HTS Code Search

This API processes shipment documents and provides intelligent HTS (Harmonized Tariff Schedule) code search using vector similarity.

## Features

- 📄 **Document Processing**: Extract information from shipping documents (PDF, Excel)
- 🔍 **HTS Code Search**: AI-powered semantic search for product classification
- 🌐 **Multilingual Support**: Search in English, Korean, or any language
- ⚡ **Fast Vector Search**: Powered by Chroma and sentence-transformers
- 🎯 **High Accuracy**: LLM-based reranking for precise results
- 🧠 **Smart Reranking**: Gemini AI validates and reorders search results

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Build Vector Store (First Time Only)

Before running the server, you need to build the HTS vector database:

```bash
python build_vector_store.py
```

This will:
- Load all HTS codes from `data/hts_json/`
- Generate embeddings using multilingual model
- Store in Chroma vector database at `data/hts_chroma/`

**Note**: First run may take 2-5 minutes to download the embedding model.

To rebuild the database:
```bash
python build_vector_store.py --rebuild
```

### 3. Run the API

```bash
uvicorn app.main:app --reload
```

or

```bash
python -m app.main
```

### 4. Access API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Document Processing
- `POST /api/process-documents`: Process shipping documents and extract information

### HTS Code Search
- `GET /api/search-hts`: Search HTS codes by product description (for short queries)
  - **Query Parameters**: 
    - `query` (string): Product description (e.g., "textile products", "섬유 제품")
    - `top_k` (int): Number of results to return (default: 5, max: 20)
  
- `POST /api/search-hts`: Search HTS codes (recommended for long product descriptions)
  - **Request Body**:
    ```json
    {
      "query": "Detailed product description...",
      "top_k": 5,
      "use_reranking": true
    }
    ```
  - **use_reranking=true**: LLM validates results (더 정확, 2-3초 추가)
  
- `GET /api/hts/{hts_code}`: Get detailed information for specific HTS code

### Health & Status
- `GET /`: API information
- `GET /api/health/vector-store`: Check vector store status

## Usage Examples

### Using curl

```bash
# Search for textile products (GET)
curl "http://localhost:8000/api/search-hts?query=textile%20products&top_k=5"

# Search in Korean (GET)
curl "http://localhost:8000/api/search-hts?query=섬유%20제품&top_k=5"

# Search with long description (POST - recommended)
curl -X POST "http://localhost:8000/api/search-hts" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Steel bracket for mounting electrical equipment on utility poles",
    "top_k": 5,
    "use_reranking": true
  }'

# Note: use_reranking=true significantly improves accuracy!

# Get specific HTS code details
curl "http://localhost:8000/api/hts/0101"
```

### Using api.http (VS Code REST Client)

Open `api.http` and click "Send Request" above any endpoint.

### Response Example

```json
{
  "query": "textile products",
  "results": [
    {
      "hts_code": "5201",
      "full_hts_code": "5201",
      "description": "Cotton, not carded or combed:",
      "similarity_score": 0.82,
      "file_path": "htsdata-2025-revision-26-5201.json"
    }
  ],
  "count": 1
}
```
## Docker (Full Stack)

### Using Docker Compose (Recommended)

Run both **frontend** and **backend** together:

1. Create a `.env` file with your API key:
```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

2. Start the full application:
```bash
docker-compose up -d
```

3. Access the application:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

4. View logs:
```bash
docker-compose logs -f
```

5. Stop the application:
```bash
docker-compose down
```

### Using Docker CLI (Backend only)

Build the Docker image: 
```bash
docker build -t document-processor .
```

Run the Docker container:
```bash
docker run -d -p 8000:8000 -e GEMINI_API_KEY=your_api_key document-processor
```

### Docker Features
- ✅ **Multi-container setup** - Frontend (Next.js) + Backend (FastAPI)
- ✅ **Multi-worker support** - 4 uvicorn workers for backend
- ✅ **Health checks** - Automatic monitoring for both services
- ✅ **Non-root user** - Enhanced security
- ✅ **Optimized builds** - Multi-stage builds for smaller images
- ✅ **Auto-restart** - Containers restart automatically on failure
- ✅ **Network isolation** - Services communicate via Docker network

**Note**: When using Docker, you'll need to build the vector store inside the container or mount it as a volume.

## Testing


Run tests:

```bash
    pytest
```

## Evaluation

Run the evaluation script:

```bash
    python evaluation.py
```

## How to run - AI Customs Tool

### Backend:
```bash
    python -m app.main
```

### Frontend:
```bash
    cd web && npm install && npm run dev
```