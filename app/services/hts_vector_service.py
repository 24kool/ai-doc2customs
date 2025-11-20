import json
from pathlib import Path
from typing import List, Dict, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
import google.generativeai as genai
from app.core.config import settings


class HTSVectorService:
    """HTS 코드 벡터 검색 서비스"""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(HTSVectorService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # 다국어 지원 임베딩 모델 (한영 혼용)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Gemini LLM for reranking
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.llm = genai.GenerativeModel("gemini-2.0-flash")
        
        self.vector_store = None
        self.persist_directory = "data/hts_chroma"
        self.data_dir = Path("data/hts_json")
        self._initialized = True
        
        print("HTSVectorService initialized")
    
    def load_hts_documents(self) -> List[Document]:
        """모든 HTS JSON 파일을 Document 객체로 로드"""
        documents = []
        
        if not self.data_dir.exists():
            print(f"Warning: {self.data_dir} directory not found")
            return documents
        
        json_files = list(self.data_dir.glob("*.json"))
        print(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # "hit" 필드가 주요 검색 대상
                hit_text = data.get("hit", "")
                if not hit_text:
                    continue
                
                # contents의 Description들도 함께 포함 (더 풍부한 컨텍스트)
                contents = data.get("contents", [])
                descriptions = []
                for item in contents:
                    desc = item.get("Description", "").strip()
                    if desc and desc not in descriptions:  # 중복 제거
                        descriptions.append(desc)
                
                # hit + 모든 descriptions를 결합
                full_text = hit_text
                if descriptions:
                    # 상위 카테고리 설명에 하위 항목들 추가
                    additional_desc = " | ".join(descriptions[:10])  # 최대 10개까지
                    full_text = f"{hit_text} || Includes: {additional_desc}"
                
                # Document 생성
                doc = Document(
                    page_content=full_text,
                    metadata={
                        "hts_code": data.get("hts_code", ""),
                        "full_hts_code": data.get("full_hts_code", ""),
                        "file_path": str(json_file.name),
                        "hit": hit_text,  # 원본 hit 저장
                        "contents_count": len(contents)
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        print(f"Successfully loaded {len(documents)} HTS documents")
        return documents
    
    def build_vector_store(self, force_rebuild: bool = False):
        """
        Vector Store 구축
        
        Args:
            force_rebuild: True면 기존 데이터 삭제 후 재구축
        """
        # 기존 컬렉션 삭제
        if force_rebuild and os.path.exists(self.persist_directory):
            import shutil
            shutil.rmtree(self.persist_directory)
            print("Removed existing vector store")
        
        # 이미 존재하면 로드
        if os.path.exists(self.persist_directory) and not force_rebuild:
            print("Vector store already exists. Loading...")
            self.load_vector_store()
            return
        
        print("Building new vector store...")
        documents = self.load_hts_documents()
        
        if not documents:
            raise ValueError("No documents found to build vector store")
        
        # Chroma Vector Store 생성
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name="hts_codes"
        )
        
        print(f"✅ Vector store built successfully with {len(documents)} documents")
        print(f"💾 Saved to: {self.persist_directory}")
        return self.vector_store
    
    def load_vector_store(self):
        """저장된 Vector Store 로드"""
        if not os.path.exists(self.persist_directory):
            raise ValueError(
                f"Vector store not found at {self.persist_directory}. "
                "Please run build_vector_store() first."
            )
        
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name="hts_codes"
        )
        
        print(f"✅ Vector store loaded from {self.persist_directory}")
        return self.vector_store
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        filter_dict: Optional[Dict] = None,
        use_reranking: bool = False
    ) -> List[Dict]:
        """
        HTS 코드 검색
        
        Args:
            query: 검색 쿼리 (예: "textile products", "섬유 제품")
            top_k: 반환할 결과 수
            filter_dict: 메타데이터 필터 (예: {"hts_code": "0101"})
            use_reranking: LLM을 사용한 재순위화 여부 (더 정확하지만 느림)
        
        Returns:
            List[Dict]: 검색 결과 리스트
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load_vector_store() first.")
        
        # 초기 검색: top_k의 2-3배 가져오기 (reranking용)
        initial_k = top_k * 3 if use_reranking else top_k
        
        # 필터링 검색
        if filter_dict:
            results = self.vector_store.similarity_search_with_score(
                query, 
                k=initial_k,
                filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search_with_score(query, k=initial_k)
        
        # 결과 포맷팅
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "hts_code": doc.metadata.get("hts_code", ""),
                "full_hts_code": doc.metadata.get("full_hts_code", ""),
                "description": doc.metadata.get("hit", doc.page_content),  # 원본 hit 표시
                "similarity_score": float(score),
                "file_path": doc.metadata.get("file_path", "")
            })
        
        # LLM Reranking
        if use_reranking and formatted_results:
            formatted_results = self._rerank_with_llm(query, formatted_results, top_k)
        
        return formatted_results[:top_k]
    
    def _rerank_with_llm(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """
        LLM을 사용하여 검색 결과를 재순위화
        
        Args:
            query: 원본 검색 쿼리
            candidates: 초기 검색 결과
            top_k: 최종 반환할 결과 수
        
        Returns:
            재순위화된 결과 리스트
        """
        try:
            # 후보들을 LLM에게 제시
            candidates_text = "\n\n".join([
                f"[{i+1}] HTS Code: {c['hts_code']}\nDescription: {c['description']}"
                for i, c in enumerate(candidates[:15])  # 최대 15개까지만
            ])
            
            prompt = f"""You are an expert in HTS (Harmonized Tariff Schedule) code classification.

Product Query:
{query}

Below are HTS code candidates. Please analyze which codes are TRULY relevant to the product described above.
Rate each code's relevance on a scale of 0-10 (10 = perfect match, 0 = completely irrelevant).

Focus on:
- Material composition (e.g., steel, plastic, textile)
- Product function and use case
- Manufacturing method
- Industry application

Candidates:
{candidates_text}

Return ONLY a JSON array with relevance scores in this exact format:
[{{"hts_code": "XXXX", "relevance": X}}, ...]

Do not include any explanation, just the JSON array."""

            response = self.llm.generate_content(prompt)
            response_text = response.text.strip()
            
            # JSON 파싱
            import re
            # Remove markdown code blocks if present
            response_text = re.sub(r'^```(json)?\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
            response_text = response_text.strip()
            
            scores = json.loads(response_text)
            
            # 점수 매핑
            score_map = {item['hts_code']: item['relevance'] for item in scores}
            
            # 후보들에 LLM 점수 추가 및 정렬
            for candidate in candidates:
                candidate['llm_relevance'] = score_map.get(candidate['hts_code'], 0)
            
            # LLM 점수로 재정렬
            reranked = sorted(candidates, key=lambda x: x.get('llm_relevance', 0), reverse=True)
            
            print(f"Reranking complete. Top result: {reranked[0]['hts_code']} (relevance: {reranked[0].get('llm_relevance', 0)})")
            
            return reranked
            
        except Exception as e:
            print(f"Reranking failed: {e}, using original order")
            return candidates
    
    def get_hts_details(self, hts_code: str) -> Optional[Dict]:
        """특정 HTS 코드의 상세 정보 가져오기"""
        json_file = self.data_dir / f"htsdata-2025-revision-26-{hts_code}.json"
        
        if not json_file.exists():
            return None
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading HTS details for {hts_code}: {e}")
            return None
    
    def add_or_update_hts(self, hts_data: Dict):
        """HTS 코드 추가 또는 업데이트 (동적 업데이트)"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        hit_text = hts_data.get("hit", "")
        if not hit_text:
            raise ValueError("'hit' field is required")
        
        doc = Document(
            page_content=hit_text,
            metadata={
                "hts_code": hts_data.get("hts_code", ""),
                "full_hts_code": hts_data.get("full_hts_code", ""),
                "file_path": f"htsdata-2025-revision-26-{hts_data.get('hts_code', '')}.json"
            }
        )
        
        # Chroma에 추가
        self.vector_store.add_documents([doc])
        print(f"✅ Added/Updated HTS code: {hts_data.get('hts_code')}")


# 싱글톤 인스턴스
hts_vector_service = HTSVectorService()

