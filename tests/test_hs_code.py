from pathlib import Path
import sys
import google.generativeai as genai
import json
import re

# Add app module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.services.document_processor import process_documents
from app.services.llm_service import extract_entity_from_document, extract_average_gross_weight_from_document, extract_average_price_from_document, extract_line_item_count_from_document

genai.configure(api_key=settings.GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

def test_extract_text(file_paths):
    # Process documents
    document_data = process_documents(file_paths)
    print("Processing Complete!")
    print(f"Extracted PDF content: {len(document_data.get('pdf_text', ''))} characters")
    print(f"Extracted Excel content: {len(document_data.get('excel_text', ''))} characters")
    
    return document_data

def extract_hs_codes(text):
    prompt = """You are a professional customs broker. Analyze all products/items in the document and find the MOST REPRESENTATIVE HS Code.

Instructions:
- Return ONLY a valid JSON object (not an array)
- Do NOT include markdown code blocks or explanations
- If multiple products exist, select the most prominent or representative one
- If HS Code is not found in document, predict the most appropriate HS Code based on product description
- For product_description, include detailed information such as: product name, material/composition, intended use, key characteristics, and any relevant specifications
- Use the following JSON structure:

{{
  "hs_code": "123456",
  "confidence": "high/medium/low",
  "product_description": "detailed product description including name, material, intended use, and key characteristics",
}}

Document text:
{text}"""
    
    response = model.generate_content(prompt.format(text=text))
    response_text = response.text.strip()
    
    # Remove markdown code blocks if present
    response_text = re.sub(r'^```(json)?\s*', '', response_text)
    response_text = re.sub(r'\s*```$', '', response_text)
    response_text = response_text.strip()
    
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw_response": response_text}

if __name__ == "__main__":
    file_paths = [
        str(Path(__file__).parent.parent / "data" / "sample2_INVOICE_CustomsFOB.pdf"),
    ]
    document_data = test_extract_text(file_paths)
    hs_codes = extract_hs_codes(document_data.get('pdf_text', ''))
    print(json.dumps(hs_codes, indent=2, ensure_ascii=False))