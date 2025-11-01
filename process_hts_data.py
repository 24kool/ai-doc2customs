#!/usr/bin/env python3
"""
HTS Data Processor for Vector Database

This script processes HTS CSV files and generates individual JSON files
for each 4-digit HTS code, suitable for vector database ingestion.

Usage:
    python process_hts_data.py <input_csv> [output_directory]

Example:
    python process_hts_data.py data/htsdata-2025-revision-26.csv
    python process_hts_data.py data/htsdata-2025-revision-26.csv data/hts_json/
"""

import csv
import json
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Any


def is_chunk_start(hts_number: str, indent: str) -> bool:
    """
    Check if this row represents the start of a new chunk.
    
    Args:
        hts_number: The HTS Number value
        indent: The Indent value
        
    Returns:
        True if this row has indent "0" and has an HTS number
    """
    # Remove quotes and whitespace
    hts_clean = hts_number.strip().strip('"')
    indent_clean = indent.strip().strip('"')
    
    # Check if indent is "0" and there's an HTS number
    return bool(indent_clean == "0" and hts_clean)


def extract_code_prefix(hts_number: str) -> str:
    """
    Extract the first 4 digits from an HTS number for file naming.
    
    Args:
        hts_number: The HTS Number value
        
    Returns:
        First 4 digits of the HTS code
    """
    # Remove quotes, whitespace, and dots
    hts_clean = hts_number.strip().strip('"').replace('.', '')
    
    # Extract first 4 digits
    match = re.match(r'^(\d{4})', hts_clean)
    if match:
        return match.group(1)
    
    # Fallback: return the cleaned number
    return hts_clean[:4] if len(hts_clean) >= 4 else hts_clean


def process_hts_csv(input_csv: str, output_dir: str) -> None:
    """
    Process HTS CSV file and generate JSON files for each 4-digit HTS code.
    
    Args:
        input_csv: Path to input CSV file
        output_dir: Directory to save output JSON files
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract base filename for output files
    input_filename = Path(input_csv).stem  # e.g., "htsdata-2025-revision-26"
    
    # Read CSV and process
    chunks: List[Dict[str, Any]] = []
    current_chunk: Dict[str, Any] = None
    current_rows: List[Dict[str, str]] = []
    
    print(f"Processing {input_csv}...")
    
    with open(input_csv, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            hts_number = row.get('HTS Number', '').strip().strip('"')
            indent = row.get('Indent', '').strip().strip('"')
            description = row.get('Description', '').strip().strip('"')
            
            # Check if this is the start of a new chunk (Indent = 0)
            if is_chunk_start(row['HTS Number'], row['Indent']):
                # Save previous chunk if it exists
                if current_chunk is not None:
                    current_chunk['contents'] = current_rows
                    chunks.append(current_chunk)
                
                # Start new chunk
                code_prefix = extract_code_prefix(hts_number)
                current_chunk = {
                    'hts_code': code_prefix,
                    'full_hts_code': hts_number,  # Keep the full code for reference
                    'hit': description
                }
                current_rows = [row]
            else:
                # Add to current chunk
                if current_chunk is not None:
                    current_rows.append(row)
        
        # Don't forget the last chunk
        if current_chunk is not None:
            current_chunk['contents'] = current_rows
            chunks.append(current_chunk)
    
    # Write JSON files
    print(f"Writing {len(chunks)} JSON files to {output_dir}...")
    
    # Track file name collisions
    filename_map = {}
    collision_count = 0
    
    for chunk in chunks:
        hts_code = chunk['hts_code']
        full_hts_code = chunk['full_hts_code']
        
        # Generate base filename
        base_filename = f"{input_filename}-{hts_code}"
        output_filename = f"{base_filename}.json"
        output_filepath = output_path / output_filename
        
        # Handle collision: if file already exists with different full_hts_code
        if output_filepath.exists():
            # Add suffix for collision
            suffix = 1
            while output_filepath.exists():
                output_filename = f"{base_filename}_{suffix}.json"
                output_filepath = output_path / output_filename
                suffix += 1
            collision_count += 1
            print(f"  ⚠ Collision detected for {hts_code} (full code: {full_hts_code}), saved as {output_filename}")
        
        # Track this filename
        filename_map[hts_code] = output_filename
        
        with open(output_filepath, 'w', encoding='utf-8') as jsonfile:
            json.dump(chunk, jsonfile, ensure_ascii=False, indent=2)
    
    print(f"✓ Successfully processed {len(chunks)} HTS code chunks")
    if collision_count > 0:
        print(f"  ⚠ {collision_count} filename collision(s) detected and resolved")
    print(f"✓ Output directory: {output_dir}")


def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print("Error: Missing required argument")
        print("\nUsage:")
        print("    python process_hts_data.py <input_csv> [output_directory]")
        print("\nExample:")
        print("    python process_hts_data.py data/htsdata-2025-revision-26.csv")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/hts_json/"
    
    # Check if input file exists
    if not os.path.exists(input_csv):
        print(f"Error: Input file '{input_csv}' not found")
        sys.exit(1)
    
    try:
        process_hts_csv(input_csv, output_dir)
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

