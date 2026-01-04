#!/usr/bin/env python
"""
EVINCE: Main Entry Point

Command-line interface for EVINCE ESG-Washing Detection.

Usage Examples:
    # Process raw OCR files into semantic chunks
    python main.py process --input data/bctn_2024_raw.txt --output data/chunks.csv
    python main.py process --input data/raw_ocr_annual_report.zip --output data/all_chunks.csv
    
    # Classify single sentence
    python main.py classify --text "Ngân hàng cam kết giảm phát thải carbon"
    
    # Classify from file
    python main.py classify --input data/sentences.csv --output results.csv
    
    # Analyze document
    python main.py analyze --input data/all_banks_sentences.csv --bank BIDV --year 2023
    
    # Generate labels with LLM
    python main.py label --input data/sentences.csv --output data/labeled.csv --sample 1000
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def process_ocr(args):
    """Process raw OCR files into semantic chunks for ESG classification."""
    from scripts.process_ocr_semantic import (
        process_single_file, 
        process_zip_file, 
        chunks_to_csv,
        MAX_TOKENS
    )
    
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
    output_path = args.output or "data/semantic_chunks.csv"
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    chunks_to_csv(chunks, output_path)
    
    # Show sample chunks
    print("\n📝 Sample chunks (with token counts):")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ({chunk.chunk_type}, {chunk.token_count} tokens, section: {chunk.section[:40] if len(chunk.section) > 40 else chunk.section}...) ---")
        print(chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text)


def classify_text(args):
    """Classify ESG topic for text."""
    from models import HuggingFaceESGClassifierInference, ESG_LABELS_VN
    
    print("Loading ESG classifier from HuggingFace...")
    classifier = HuggingFaceESGClassifierInference(device=args.device)
    print("Model loaded!\n")
    
    if args.text:
        # Single text classification
        result = classifier.predict(args.text)
        print(f"Text: {args.text}")
        print(f"─" * 50)
        print(f"Label: {result.predicted_label}")
        print(f"Label (VN): {result.predicted_label_vn}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Is ESG: {result.is_esg}")
        print(f"\nProbabilities:")
        for label, prob in sorted(result.probabilities.items(), key=lambda x: -x[1]):
            vn_name = ESG_LABELS_VN.get(label, label)
            print(f"  {label}: {prob:.2%} ({vn_name})")
    
    elif args.input:
        # Batch classification from file
        print(f"Loading data from: {args.input}")
        df = pd.read_csv(args.input)
        
        # Support both 'sentence' and 'text' column names
        text_col = 'sentence' if 'sentence' in df.columns else 'text'
        if text_col not in df.columns:
            print("Error: CSV must have 'sentence' or 'text' column")
            return
        
        texts = df[text_col].tolist()
        print(f"Classifying {len(texts)} texts...")
        
        results = classifier.predict_batch(texts, batch_size=args.batch_size)
        
        # Add results to dataframe
        df['esg_label'] = [r.predicted_label for r in results]
        df['esg_label_vn'] = [r.predicted_label_vn for r in results]
        df['esg_confidence'] = [r.confidence for r in results]
        df['is_esg'] = [r.is_esg for r in results]
        
        # Save output
        output_path = args.output or args.input.replace('.csv', '_classified.csv')
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\nResults saved to: {output_path}")
        
        # Print summary
        print(f"\nLabel Distribution:")
        print(df['esg_label'].value_counts())
    else:
        print("Error: Provide --text or --input")


def analyze_document(args):
    """Analyze document for ESG-washing."""
    from claim_evidence import DocumentAnalyzer
    
    print("Loading Document Analyzer...")
    analyzer = DocumentAnalyzer(device=args.device)
    print("Analyzer loaded!\n")
    
    if args.input:
        print(f"Loading data from: {args.input}")
        df = pd.read_csv(args.input)
        
        # Support both 'sentence' and 'text' column names
        text_col = 'sentence' if 'sentence' in df.columns else 'text'
        
        # Filter by bank/year if specified
        if args.bank:
            df = df[df['bank'].str.lower() == args.bank.lower()]
        if args.year:
            df = df[df['year'] == args.year]
        
        if len(df) == 0:
            print("No sentences found with given filters")
            return
        
        sentences = df[text_col].tolist()
        print(f"Analyzing {len(sentences)} sentences...")
        
        result = analyzer.analyze_document(
            sentences=sentences,
            bank=args.bank or "Unknown",
            year=args.year or 0
        )
        
        print(f"\n{'='*60}")
        print(f"DOCUMENT ANALYSIS RESULT")
        print(f"{'='*60}")
        print(f"Bank: {args.bank or 'All'}")
        print(f"Year: {args.year or 'All'}")
        print(f"Total Sentences: {len(sentences)}")
        print(f"─" * 60)
        print(f"Document Washing Index: {result.document_washing_index:.3f}")
        print(f"Total Claims: {result.total_claims}")
        print(f"Verified Claims: {result.verified_claims}")
        print(f"High Risk Claims: {result.high_risk_claims}")
        print(f"Average Verification Score: {result.avg_verification_score:.3f}")
        print(f"{'='*60}")
        
        if args.output:
            # Save detailed results
            import json
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"\nDetailed results saved to: {args.output}")
    else:
        print("Error: Provide --input")


def generate_labels(args):
    """Generate labels using LLM."""
    from scripts.llm_labeling import run_labeling
    
    if not args.input:
        print("Error: Provide --input")
        return
    
    output = args.output or args.input.replace('.csv', '_labeled.csv')
    
    run_labeling(
        input_path=args.input,
        output_path=output,
        sample_size=args.sample,
        resume=not args.no_resume,
        workers=args.workers
    )


def interactive_mode(args):
    """Interactive classification mode."""
    from models import HuggingFaceESGClassifierInference, ESG_LABELS_VN
    
    print("Loading ESG classifier...")
    classifier = HuggingFaceESGClassifierInference(device=args.device)
    print("Model loaded! Enter sentences to classify (Ctrl+C to exit)\n")
    
    try:
        while True:
            text = input(">>> ").strip()
            if not text:
                continue
            
            result = classifier.predict(text)
            print(f"  → {result.predicted_label} ({result.confidence:.1%})")
            print(f"    {result.predicted_label_vn}")
            print()
    except KeyboardInterrupt:
        print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="EVINCE: ESG-Washing Detection for Vietnamese Banking Reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process raw OCR files into semantic chunks
  python main.py process -i data/bctn_2024_raw.txt -o data/chunks.csv
  
  # Interactive mode
  python main.py interactive
  
  # Classify single text
  python main.py classify -t "Ngân hàng cam kết giảm phát thải carbon"
  
  # Classify CSV file
  python main.py classify -i data/chunks.csv -o results.csv
  
  # Analyze document for washing
  python main.py analyze -i data/chunks.csv --bank agribank --year 2024
  
  # Generate labels with LLM
  python main.py label -i data/chunks.csv -o labeled.csv --sample 1000
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Process command (NEW)
    process_parser = subparsers.add_parser("process", help="Process raw OCR files into semantic chunks")
    process_parser.add_argument("-i", "--input", type=str, required=True, help="Input file (txt or zip)")
    process_parser.add_argument("-o", "--output", type=str, default="data/semantic_chunks.csv", help="Output CSV file")
    
    # Classify command
    classify_parser = subparsers.add_parser("classify", help="Classify ESG topics")
    classify_parser.add_argument("-t", "--text", type=str, help="Single text to classify")
    classify_parser.add_argument("-i", "--input", type=str, help="Input CSV file")
    classify_parser.add_argument("-o", "--output", type=str, help="Output CSV file")
    classify_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    classify_parser.add_argument("--device", type=str, default="auto", help="Device: cpu, cuda, auto")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze document for washing")
    analyze_parser.add_argument("-i", "--input", type=str, required=True, help="Input CSV file")
    analyze_parser.add_argument("-o", "--output", type=str, help="Output JSON file")
    analyze_parser.add_argument("--bank", type=str, help="Filter by bank name")
    analyze_parser.add_argument("--year", type=int, help="Filter by year")
    analyze_parser.add_argument("--device", type=str, default="auto", help="Device: cpu, cuda, auto")
    
    # Label command
    label_parser = subparsers.add_parser("label", help="Generate labels with LLM")
    label_parser.add_argument("-i", "--input", type=str, required=True, help="Input CSV file")
    label_parser.add_argument("-o", "--output", type=str, help="Output CSV file")
    label_parser.add_argument("--sample", type=int, default=None, help="Number of samples (default: all)")
    label_parser.add_argument("--workers", "-w", type=int, default=1, help="Number of parallel workers")
    label_parser.add_argument("--no-resume", action="store_true", help="Don't resume from previous run")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    interactive_parser.add_argument("--device", type=str, default="auto", help="Device: cpu, cuda, auto")
    
    args = parser.parse_args()
    
    if args.command == "process":
        process_ocr(args)
    elif args.command == "classify":
        classify_text(args)
    elif args.command == "analyze":
        analyze_document(args)
    elif args.command == "label":
        generate_labels(args)
    elif args.command == "interactive":
        interactive_mode(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

