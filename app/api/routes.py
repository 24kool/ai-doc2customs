from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Body
from typing import List, Optional
from pydantic import BaseModel, Field
import os
import tempfile

from app.services.document_processor import process_documents
from app.services.llm_service import extract_entity_from_document, extract_average_gross_weight_from_document, extract_average_price_from_document, extract_line_item_count_from_document
from app.services.hts_vector_service import hts_vector_service

router = APIRouter()


class HTSSearchRequest(BaseModel):
    """HTS 코드 검색 요청 모델"""
    query: str = Field(..., description="검색 쿼리 (제품 설명)")
    top_k: int = Field(5, ge=1, le=20, description="반환할 결과 수")
    use_reranking: bool = Field(False, description="LLM 기반 재순위화 사용 (더 정확하지만 느림)")

@router.post("/process-documents", response_model=dict)
async def process_documents_endpoint(
    files: List[UploadFile] = File(...)
):
    temp_file_paths = []
    for file in files:
        # Save uploaded file temporarily with correct extension
        file_extension = os.path.splitext(file.filename)[1] if file.filename else ''
        temp_file = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)
        temp_file_paths.append(temp_file.name)

        # Write content to temp file
        content = await file.read()
        temp_file.write(content)
        temp_file.close()

    # Process documents
    document_data = process_documents(temp_file_paths)

    # Extract data from document
    general_entity = extract_entity_from_document(document_data)

    # Extract average numbers from document
    avg_numbers_list, avg_numbers = extract_average_gross_weight_from_document(document_data)

    price_list, average_price = extract_average_price_from_document(document_data)
    
    line_item_count = extract_line_item_count_from_document(document_data)

    # Clean up temp files
    for path in temp_file_paths:
        os.unlink(path)

    return {
        "general_entity": general_entity,
        "gross_weight_list": avg_numbers_list,
        "average_gross_weight": avg_numbers,
        "price_list": price_list,
        "average_price": average_price,
        "line_item_count": line_item_count
    }


@router.get("/search-hts", response_model=dict)
async def search_hts_codes_get(
    query: str = Query(..., description="검색 쿼리 (예: 'textile products', '섬유 제품')"),
    top_k: int = Query(5, ge=1, le=20, description="반환할 결과 수"),
    use_reranking: bool = Query(False, description="LLM 재순위화 사용 (더 정확)")
):
    """
    HTS 코드 벡터 검색 (GET 메소드)
    
    짧은 쿼리에 적합합니다. 긴 제품 설명은 POST /search-hts를 사용하세요.
    
    - query: 검색하고자 하는 제품 설명 (자연어)
    - top_k: 반환할 유사 HTS 코드 수
    - use_reranking: LLM으로 결과 재평가 (더 정확하지만 2-3초 소요)
    """
    try:
        results = hts_vector_service.search(query, top_k=top_k, use_reranking=use_reranking)
        
        return {
            "query": query,
            "results": results,
            "count": len(results),
            "reranked": use_reranking
        }
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/search-hts", response_model=dict)
async def search_hts_codes_post(request: HTSSearchRequest):
    """
    HTS 코드 벡터 검색 (POST 메소드)
    
    긴 제품 설명이나 복잡한 쿼리에 적합합니다.
    
    Request Body:
    - query: 검색하고자 하는 제품 설명 (자연어, 제한 없음)
    - top_k: 반환할 유사 HTS 코드 수 (기본값: 5, 최대: 20)
    - use_reranking: LLM으로 결과 재평가 (기본값: false)
    
    Example:
    ```json
    {
        "query": "Bracket, Secondary Rack, Extended. Made from steel...",
        "top_k": 5,
        "use_reranking": true
    }
    ```
    
    **use_reranking=true** 권장: 더 정확한 결과를 얻을 수 있습니다 (2-3초 추가 소요)
    """
    try:
        results = hts_vector_service.search(
            request.query, 
            top_k=request.top_k,
            use_reranking=request.use_reranking
        )
        
        return {
            "query": request.query,
            "results": results,
            "count": len(results),
            "reranked": request.use_reranking
        }
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/hts/{hts_code}", response_model=dict)
async def get_hts_details(hts_code: str):
    """
    특정 HTS 코드의 상세 정보 조회
    
    - hts_code: HTS 코드 (예: '0101')
    """
    details = hts_vector_service.get_hts_details(hts_code)
    
    if details is None:
        raise HTTPException(status_code=404, detail=f"HTS code '{hts_code}' not found")
    
    return details


@router.get("/health/vector-store", response_model=dict)
async def check_vector_store_health():
    """Vector Store 상태 확인"""
    try:
        if hts_vector_service.vector_store is None:
            return {
                "status": "not_initialized",
                "message": "Vector store not initialized"
            }
        
        # 간단한 테스트 검색
        test_results = hts_vector_service.search("test", top_k=1)
        
        return {
            "status": "healthy",
            "message": "Vector store is working",
            "sample_count": len(test_results)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        } 
