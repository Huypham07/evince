
import os
import re
import zipfile
import pandas as pd
from pathlib import Path
import argparse

def extract_metadata(filename):
    """
    Extract bank, year, and report type from filename.
    Expected format: bctn_2015_raw.txt inside a folder like 'agribank'
    """
    # Normalize path separators
    filename = filename.replace('\\', '/')
    parts = filename.split('/')
    
    bank = "unknown"
    if len(parts) > 1:
        # parent folder usually is bank name
        bank = parts[-2]
    
    basename = parts[-1]
    
    # Try to find year
    year_match = re.search(r'20\d{2}', basename)
    year = int(year_match.group(0)) if year_match else 0
    
    # Report type
    if "bctn" in basename.lower():
        report_type = "BCTN" # Annual Report
    elif "bctdml" in basename.lower() or "esg" in basename.lower():
        report_type = "BCTDML" # Sustainability Report
    else:
        report_type = "OTHER"
        
    return bank, year, report_type

def split_sentences(text):
    """
    Simple sentence splitting for Vietnamese.
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split by punctuation followed by space and uppercase or end of line
    # This is a heuristic; for production use a library like PyVi or Underthesea
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZĂÂÁÀẢÃẠĂẰẲẴẶÂẦẨẪẬĐEÊÉÈẺẼẸÊỀỂỄỆIÍÌỈĨỊOÔƠÓÒỎÕỌÔỒỔỖỘƠỜỞỠỢUƯÚÙỦŨỤƯỪỬỮỰYÝỲỶỸỴ])', text)
    
    return [s.strip() for s in sentences if len(s.strip()) > 10]

def process_ocr_zip(zip_path, output_path):
    print(f"Processing {zip_path}...")
    
    data = []
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        for file_info in z.infolist():
            if file_info.is_dir() or not file_info.filename.endswith('.txt') or "__MACOSX" in file_info.filename:
                continue
                
            try:
                with z.open(file_info) as f:
                    content = f.read().decode('utf-8')
                    
                bank, year, report_type = extract_metadata(file_info.filename)
                sentences = split_sentences(content)
                
                print(f"  Parses {file_info.filename}: Bank={bank}, Year={year}, Type={report_type}, Sentences={len(sentences)}")
                
                for s in sentences:
                    data.append({
                        "sentence": s,
                        "bank": bank,
                        "year": year,
                        "report_type": report_type,
                        "source_file": file_info.filename
                    })
                    
            except Exception as e:
                print(f"  Error processing {file_info.filename}: {e}")
                
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Saved {len(df)} sentences to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process OCR Zip file to Sentences CSV")
    parser.add_argument("--input", type=str, default="data/raw_ocr_annual_report.zip", help="Path to zip file")
    parser.add_argument("--output", type=str, default="data/all_banks_sentences.csv", help="Path to output CSV")
    
    args = parser.parse_args()
    process_ocr_zip(args.input, args.output)
