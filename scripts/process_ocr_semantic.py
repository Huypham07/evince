#!/usr/bin/env python3
"""
EVINCE: Smart OCR Processor with Semantic Chunking

Processes raw OCR text files (Markdown format) into semantic chunks
that preserve context for ESG classification.

Features:
- Detects sections using Markdown headers (##)
- Filters noise (page numbers, image placeholders, short lines)
- Groups consecutive paragraphs within sections
- Preserves tables as context
- Uses PhoBERT tokenizer to ensure chunks fit 512 token limit
- Outputs CSV with semantic chunks ready for ESG classification

Usage:
    python scripts/process_ocr_semantic.py --input data/bctn_2024_raw.txt --output data/chunks.csv
    python scripts/process_ocr_semantic.py --input data/raw_ocr_annual_report.zip --output data/all_chunks.csv
"""

import argparse
import re
import os
import zipfile
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

# Try to import tokenizer for accurate token counting
try:
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    TOKENIZER_AVAILABLE = True
    print("✓ PhoBERT tokenizer loaded for accurate token counting")
except Exception as e:
    TOKENIZER = None
    TOKENIZER_AVAILABLE = False
    print(f"⚠ Tokenizer not available, using character-based estimation: {e}")


# Token limit for PhoBERT
MAX_TOKENS = 500  # Leave some margin for special tokens
MAX_CHARS_FALLBACK = 800  # Conservative estimate when tokenizer not available


@dataclass
class SemanticChunk:
    """A semantic unit of text with context."""
    text: str
    section: str  # e.g., "THÔNG ĐIỆP CỦA BAN LÃNH ĐẠO AGRIBANK"
    chunk_type: str  # "paragraph", "table", "list"
    source_file: str
    bank: str = ""
    year: int = 0
    report_type: str = ""
    start_line: int = 0
    end_line: int = 0
    token_count: int = 0


def count_tokens(text: str) -> int:
    """Count tokens using PhoBERT tokenizer or estimate from characters."""
    if TOKENIZER_AVAILABLE and TOKENIZER:
        return len(TOKENIZER.encode(text, add_special_tokens=False))
    else:
        # Rough estimate: Vietnamese text averages ~2.5 chars per token
        return len(text) // 2


def split_text_by_tokens(text: str, max_tokens: int = MAX_TOKENS) -> List[str]:
    """
    Split text into chunks that fit within token limit.
    Tries to split at sentence boundaries for coherence.
    """
    if count_tokens(text) <= max_tokens:
        return [text]
    
    # Split by sentences (Vietnamese sentence endings)
    sentence_pattern = r'(?<=[.!?。])\s+'
    sentences = re.split(sentence_pattern, text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_tokens = count_tokens(sentence)
        
        # If single sentence exceeds limit, split by characters
        if sentence_tokens > max_tokens:
            # Save current chunk first
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            # Split long sentence by half recursively
            words = sentence.split()
            mid = len(words) // 2
            first_half = " ".join(words[:mid])
            second_half = " ".join(words[mid:])
            
            for part in [first_half, second_half]:
                if count_tokens(part) <= max_tokens:
                    chunks.append(part)
                else:
                    # Still too long, just truncate
                    if TOKENIZER_AVAILABLE and TOKENIZER:
                        tokens = TOKENIZER.encode(part, add_special_tokens=False)[:max_tokens]
                        chunks.append(TOKENIZER.decode(tokens))
                    else:
                        chunks.append(part[:MAX_CHARS_FALLBACK])
        
        elif current_tokens + sentence_tokens > max_tokens:
            # Save current chunk and start new one
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks if chunks else [text[:MAX_CHARS_FALLBACK]]


def is_noise_line(line: str) -> bool:
    """Check if a line is noise (page number, image placeholder, etc.)."""
    line = line.strip()
    
    # Empty line
    if not line:
        return True
    
    # Image placeholder
    if "<!-- image -->" in line:
        return True
    
    # Page number only (1-3 digits)
    if re.match(r'^\d{1,3}$', line):
        return True
    
    # Report header/footer
    if "BÁO CÁO THƯỜNG NIÊN" in line and len(line) < 50:
        return True
    
    # Too short to be meaningful (less than 10 chars after stripping)
    if len(line) < 10:
        return True
    
    # Only special characters
    if re.match(r'^[\s\-\|\_\=\*\#\.\:\,]+$', line):
        return True
    
    return False


def is_header_line(line: str) -> bool:
    """Check if line is a section header."""
    return line.strip().startswith("##")


def is_table_line(line: str) -> bool:
    """Check if line is part of a Markdown table."""
    line = line.strip()
    return bool(re.match(r'^\|.*\|$', line)) or bool(re.match(r'^\|[\-\|]+\|$', line))


def extract_section_title(line: str) -> str:
    """Extract section title from header line."""
    # Remove ## prefix
    title = re.sub(r'^#+\s*', '', line.strip())
    # Clean up common artifacts
    title = title.strip()
    return title


def extract_metadata_from_filename(filename: str) -> Tuple[str, int, str]:
    """
    Extract bank, year, and report type from filename.
    
    Examples:
        raw_ocr_annual_report/agribank/bctn_2015_raw.txt -> ("agribank", 2015, "bctn")
        bctn_2024_raw.txt -> ("unknown", 2024, "bctn")
    """
    # Try to extract year
    year_match = re.search(r'(\d{4})', filename)
    year = int(year_match.group(1)) if year_match else 0
    
    # Try to extract bank from path
    bank = "unknown"
    parts = Path(filename).parts
    for part in parts:
        if part.lower() not in ['data', 'raw_ocr_annual_report', 'raw', ''] and not re.match(r'.*\d{4}.*', part):
            bank = part.lower()
            break
    
    # Try to extract report type
    report_type = "unknown"
    if "bctn" in filename.lower():
        report_type = "bctn"
    elif "bptn" in filename.lower():
        report_type = "bptn"
    elif "esg" in filename.lower():
        report_type = "esg_report"
    
    return bank, year, report_type


def create_semantic_chunks(lines: List[str], source_file: str) -> List[SemanticChunk]:
    """
    Process lines and create semantic chunks with token limit enforcement.
    
    Strategy:
    1. Identify section headers (##)
    2. Group consecutive non-noise lines within each section
    3. Treat tables as single chunks
    4. Split chunks that exceed 512 tokens
    """
    chunks = []
    current_section = "UNKNOWN"
    current_chunk_lines = []
    current_chunk_start = 0
    in_table = False
    table_lines = []
    table_start = 0
    
    bank, year, report_type = extract_metadata_from_filename(source_file)
    
    def save_chunk_with_token_check(text: str, chunk_type: str, start_line: int, end_line: int):
        """Save chunk, splitting if necessary to fit token limit."""
        text = text.strip()
        if len(text) < 30:  # Minimum meaningful chunk
            return
        
        # Split if exceeds token limit
        text_parts = split_text_by_tokens(text, MAX_TOKENS)
        
        for part in text_parts:
            part = part.strip()
            if len(part) >= 30:
                token_count = count_tokens(part)
                chunks.append(SemanticChunk(
                    text=part,
                    section=current_section,
                    chunk_type=chunk_type,
                    source_file=source_file,
                    bank=bank,
                    year=year,
                    report_type=report_type,
                    start_line=start_line,
                    end_line=end_line,
                    token_count=token_count
                ))
    
    def save_current_chunk():
        """Save the current accumulated chunk."""
        nonlocal current_chunk_lines, current_chunk_start
        
        if current_chunk_lines:
            text = "\n".join(current_chunk_lines)
            save_chunk_with_token_check(
                text, "paragraph",
                current_chunk_start,
                current_chunk_start + len(current_chunk_lines) - 1
            )
            current_chunk_lines = []
    
    def save_table_chunk():
        """Save the current table as a chunk."""
        nonlocal table_lines, table_start, in_table
        
        if table_lines:
            text = "\n".join(table_lines)
            save_chunk_with_token_check(
                text, "table",
                table_start,
                table_start + len(table_lines) - 1
            )
            table_lines = []
            in_table = False
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Skip noise
        if is_noise_line(line):
            # If we were building a table, finalize it
            if in_table:
                save_table_chunk()
            continue
        
        # Handle section header
        if is_header_line(line_stripped):
            # Save any pending chunks
            save_current_chunk()
            if in_table:
                save_table_chunk()
            
            # Update section
            current_section = extract_section_title(line_stripped)
            current_chunk_start = i + 1  # Next content starts after header
            continue
        
        # Handle table
        if is_table_line(line_stripped):
            # Save pending paragraph
            save_current_chunk()
            
            if not in_table:
                in_table = True
                table_start = i
            table_lines.append(line_stripped)
            continue
        else:
            # If we were in a table, save it
            if in_table:
                save_table_chunk()
        
        # Regular paragraph content
        if not current_chunk_lines:
            current_chunk_start = i
        
        current_chunk_lines.append(line_stripped)
    
    # Save any remaining chunks
    save_current_chunk()
    if in_table:
        save_table_chunk()
    
    return chunks


def process_single_file(file_path: str) -> List[SemanticChunk]:
    """Process a single OCR text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    return create_semantic_chunks(lines, file_path)


def process_zip_file(zip_path: str) -> List[SemanticChunk]:
    """Process all text files in a zip archive."""
    all_chunks = []
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for file_info in zf.infolist():
            if file_info.filename.endswith('.txt') and not file_info.is_dir():
                print(f"  Processing: {file_info.filename}")
                
                with zf.open(file_info.filename) as f:
                    content = f.read().decode('utf-8')
                    lines = content.split('\n')
                
                chunks = create_semantic_chunks(lines, file_info.filename)
                all_chunks.extend(chunks)
                print(f"    -> {len(chunks)} chunks extracted")
    
    return all_chunks


def chunks_to_csv(chunks: List[SemanticChunk], output_path: str):
    """Save chunks to CSV file."""
    import pandas as pd
    
    data = []
    for chunk in chunks:
        data.append({
            "text": chunk.text,
            "section": chunk.section,
            "chunk_type": chunk.chunk_type,
            "bank": chunk.bank,
            "year": chunk.year,
            "report_type": chunk.report_type,
            "source_file": chunk.source_file,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "token_count": chunk.token_count,
            "char_count": len(chunk.text)
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nSaved {len(df)} chunks to: {output_path}")
    
    # Print statistics
    print("\n📊 Statistics:")
    print(f"  Total chunks: {len(df)}")
    print(f"  Paragraph chunks: {len(df[df['chunk_type'] == 'paragraph'])}")
    print(f"  Table chunks: {len(df[df['chunk_type'] == 'table'])}")
    print(f"  Average token count: {df['token_count'].mean():.0f}")
    print(f"  Max token count: {df['token_count'].max()}")
    print(f"  Chunks over 500 tokens: {len(df[df['token_count'] > 500])}")
    print(f"  Unique sections: {df['section'].nunique()}")
    
    if 'bank' in df.columns:
        print(f"  Banks: {df['bank'].unique().tolist()}")
    if 'year' in df.columns:
        years = df[df['year'] > 0]['year'].unique().tolist()
        print(f"  Years: {sorted(years)}")


def main():
    parser = argparse.ArgumentParser(
        description="Process OCR text files into semantic chunks for ESG classification"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input file (txt or zip)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/semantic_chunks.csv",
        help="Output CSV file (default: data/semantic_chunks.csv)"
    )
    
    args = parser.parse_args()
    
    print(f"🔍 Processing: {args.input}")
    print(f"📏 Token limit: {MAX_TOKENS}")
    
    if args.input.endswith('.zip'):
        chunks = process_zip_file(args.input)
    elif args.input.endswith('.txt'):
        chunks = process_single_file(args.input)
    else:
        print(f"Error: Unsupported file format. Use .txt or .zip")
        return
    
    if not chunks:
        print("Warning: No chunks extracted!")
        return
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    chunks_to_csv(chunks, args.output)
    
    # Show sample chunks
    print("\n📝 Sample chunks (with token counts):")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ({chunk.chunk_type}, {chunk.token_count} tokens, section: {chunk.section[:40]}...) ---")
        print(chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text)


if __name__ == "__main__":
    main()
