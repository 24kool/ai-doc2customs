"""
HTS Vector Store 구축 스크립트

이 스크립트는 data/hts_json 폴더의 JSON 파일들을 읽어서
Chroma 기반 벡터 데이터베이스를 구축합니다.

Usage:
    python build_vector_store.py [--rebuild]
    
Options:
    --rebuild    기존 벡터 스토어를 삭제하고 새로 구축
"""

import sys
import argparse
from app.services.hts_vector_service import hts_vector_service


def main():
    parser = argparse.ArgumentParser(
        description="Build HTS Vector Store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 처음 구축하기
    python build_vector_store.py
    
    # 기존 데이터 삭제 후 재구축
    python build_vector_store.py --rebuild
        """
    )
    parser.add_argument(
        "--rebuild", 
        action="store_true", 
        help="Force rebuild (delete existing and recreate)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("HTS Vector Store Builder")
    print("=" * 70)
    print()
    
    if args.rebuild:
        print("⚠️  Rebuild mode: Existing vector store will be deleted")
        print()
    
    try:
        print("📦 Loading HTS JSON files...")
        print("🔨 Building vector embeddings...")
        print("   (This may take a few minutes on first run)")
        print()
        
        hts_vector_service.build_vector_store(force_rebuild=args.rebuild)
        
        print()
        print("=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print()
        print("Vector store is ready. You can now:")
        print("  1. Start the server:")
        print("     uvicorn app.main:app --reload")
        print()
        print("  2. Test the API:")
        print("     http://localhost:8000/docs")
        print()
        print("  3. Example search:")
        print('     curl "http://localhost:8000/api/search-hts?query=animals&top_k=5"')
        print()
        
    except Exception as e:
        print()
        print("=" * 70)
        print("❌ FAILED!")
        print("=" * 70)
        print()
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Make sure data/hts_json directory exists and contains JSON files")
        print("  2. Install required packages: pip install -r requirements.txt")
        print("  3. Check if you have enough disk space")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()


