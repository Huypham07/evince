"""
EVINCE: LLM Labeling Script

Generate training labels using LLM (Qwen3, AWS Bedrock, or Gemini) for:
1. ESG Topic Classification (E, S, G, Financing, Policy, Non-ESG)
2. Sentence Type Classification (CLAIM, EVIDENCE, CONTEXT, NON-ESG)
3. Washing Detection (7 washing types)

Usage:
    python -m evince.scripts.llm_labeling --sample_size 1000 --output labeled_data.csv
    python main.py label --input data/chunks.csv --workers 4
    
Environment Variables:
    LLM_PROVIDER: "qwen", "bedrock", or "gemini" (default: gemini)
    AWS_BEDROCK_REGION: AWS region for Bedrock (default: us-east-1)
    BEDROCK_MODEL: Model alias (default: claude-haiku)
"""

import os
import sys
import json
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Config


# Prompts for labeling
ESG_TOPIC_PROMPT = """Bạn là chuyên gia phân tích ESG (Môi trường, Xã hội, Quản trị) cho ngân hàng Việt Nam.

Phân loại câu sau vào MỘT trong các nhãn:
- E: Môi trường (CO2, năng lượng, rác thải, khí hậu, xanh)
- S: Xã hội (nhân viên, cộng đồng, sức khỏe, an toàn, đào tạo)
- G: Quản trị (hội đồng, đạo đức, tuân thủ, rủi ro, minh bạch)
- Financing: Tín dụng xanh, trái phiếu ESG, cho vay bền vững
- Policy: Chính sách nội bộ, cam kết, quy định
- Non-ESG: Không liên quan ESG (báo cáo tài chính, hành chính)

Câu: "{sentence}"

Trả lời JSON: {{"label": "...", "confidence": 0.0-1.0, "reason": "..."}}
"""

SENTENCE_TYPE_PROMPT = """Phân loại câu ESG sau vào một trong các loại:
- CLAIM: Cam kết, hứa hẹn, mục tiêu (có từ: cam kết, sẽ, hướng tới, mục tiêu, phấn đấu)
- EVIDENCE: Bằng chứng, số liệu cụ thể, hành động đã thực hiện (có từ: đã, hoàn thành, năm 20xx, %, số liệu)
- CONTEXT: Thông tin nền, giải thích (không phải cam kết, không có số liệu)
- NON_ESG: Không liên quan ESG

Câu: "{sentence}"

Trả lời JSON: {{"type": "...", "confidence": 0.0-1.0}}
"""

WASHING_DETECTION_PROMPT = """Bạn là chuyên gia phát hiện ESG-Washing (tẩy xanh) trong báo cáo ngân hàng.

Phân tích câu sau và xác định loại washing (nếu có):
- NOT_WASHING: Cam kết/tuyên bố genuine, có bằng chứng rõ ràng
- VAGUE_COMMITMENT: Cam kết mơ hồ, không có số liệu cụ thể
- SELECTIVE_DISCLOSURE: Chỉ nêu điểm tốt, giấu điểm xấu
- SYMBOLIC_ACTION: Hành động mang tính biểu tượng (giải thưởng, chứng nhận) nhưng không có nội dung thực chất
- DECOUPLING: Nói một đằng làm một nẻo
- MISLEADING_METRICS: Số liệu gây hiểu lầm (% tương đối, không có baseline)
- FUTURE_DEFLECTION: Trì hoãn sang tương lai (sẽ, dự kiến, kỳ vọng)

Câu: "{sentence}"

Trả lời JSON: {{"washing_type": "...", "confidence": 0.0-1.0, "reason": "..."}}
"""


def get_llm_client(provider: str = None):
    """
    Get LLM client based on provider setting.
    
    Args:
        provider: "qwen", "bedrock", or "gemini". Uses Config.LLM_PROVIDER if not specified.
        
    Returns:
        LLM client instance
    """
    provider = provider or Config.LLM_PROVIDER
    
    if provider == "bedrock":
        from core import BedrockClient
        return BedrockClient(
            region=Config.AWS_BEDROCK_REGION,
            model_id=Config.BEDROCK_MODEL,
            max_tokens=Config.BEDROCK_MAX_TOKENS
        )
    elif provider == "gemini":
        from core import GeminiClient
        return GeminiClient()
    elif provider == "qwen":
        from core import QwenClient
        return QwenClient()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'qwen', 'bedrock', or 'gemini'.")


class ESGLabeler:
    """LLM-based labeler for ESG data. Supports Qwen, Bedrock, and Gemini."""
    
    def __init__(self, provider: str = None, worker_id: int = None):
        """
        Initialize labeler with specified LLM provider.
        
        Args:
            provider: "qwen", "bedrock", or "gemini". Uses Config.LLM_PROVIDER if not specified.
            worker_id: Worker ID for parallel processing (for logging)
        """
        self.provider = provider or Config.LLM_PROVIDER
        self.client = get_llm_client(self.provider)
        self.worker_id = worker_id
        if worker_id is not None:
            print(f"Worker {worker_id}: Initialized ESGLabeler with provider: {self.provider}")
        else:
            print(f"Initialized ESGLabeler with provider: {self.provider}")
    
    def label_esg_topic(self, sentence: str) -> Dict:
        """Label ESG topic."""
        prompt = ESG_TOPIC_PROMPT.format(sentence=sentence)
        result = self.client.generate_content(prompt, temperature=0.0)
        
        if isinstance(result, dict) and "label" in result:
            return result
        return {"label": "Non-ESG", "confidence": 0.0, "error": str(result)}
    
    def label_sentence_type(self, sentence: str) -> Dict:
        """Label sentence type."""
        prompt = SENTENCE_TYPE_PROMPT.format(sentence=sentence)
        result = self.client.generate_content(prompt, temperature=0.0)
        
        if isinstance(result, dict) and "type" in result:
            return result
        return {"type": "CONTEXT", "confidence": 0.0, "error": str(result)}
    
    def detect_washing(self, sentence: str) -> Dict:
        """Detect washing type."""
        prompt = WASHING_DETECTION_PROMPT.format(sentence=sentence)
        result = self.client.generate_content(prompt, temperature=0.0)
        
        if isinstance(result, dict) and "washing_type" in result:
            return result
        return {"washing_type": "NOT_WASHING", "confidence": 0.0, "error": str(result)}
    
    def label_full(self, sentence: str) -> Dict:
        """Full labeling: ESG topic + sentence type + washing."""
        esg = self.label_esg_topic(sentence)
        
        result = {
            "sentence": sentence,
            "esg_label": esg.get("label", "Non-ESG"),
            "esg_confidence": esg.get("confidence", 0.0),
            "esg_reason": esg.get("reason", ""),
        }
        
        # Only label further if ESG-related
        if result["esg_label"] != "Non-ESG":
            sent_type = self.label_sentence_type(sentence)
            washing = self.detect_washing(sentence)
            
            result.update({
                "sentence_type": sent_type.get("type", "CONTEXT"),
                "sentence_type_confidence": sent_type.get("confidence", 0.0),
                "washing_type": washing.get("washing_type", "NOT_WASHING"),
                "washing_confidence": washing.get("confidence", 0.0),
                "washing_reason": washing.get("reason", ""),
            })
        else:
            result.update({
                "sentence_type": "NON_ESG",
                "sentence_type_confidence": 1.0,
                "washing_type": "NOT_WASHING",
                "washing_confidence": 1.0,
                "washing_reason": "Non-ESG sentence",
            })
        
        return result


def load_data(csv_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load sentences from CSV. Supports both 'text' and 'sentence' columns."""
    df = pd.read_csv(csv_path)
    
    # Determine text column name (support both new 'text' and legacy 'sentence')
    text_col = 'text' if 'text' in df.columns else 'sentence'
    
    # Filter short/noisy sentences
    df = df[df[text_col].str.len() > 20]
    df = df[df[text_col].str.split().str.len() > 5]
    
    # Normalize column name to 'text' for consistency
    if text_col == 'sentence':
        df = df.rename(columns={'sentence': 'text'})
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    return df.reset_index(drop=True)


def label_single_row(row: pd.Series, labeler: ESGLabeler, labeled_texts: set) -> Optional[Dict]:
    """Label a single row and return result."""
    text = row['text']
    
    if text in labeled_texts:
        return None
    
    try:
        result = labeler.label_full(text)
        result.update({
            "text": text,
            "bank": row.get('bank', ''),
            "year": row.get('year', 0),
            "report_type": row.get('report_type', ''),
        })
        return result
    except Exception as e:
        print(f"Error labeling: {e}")
        return None


def worker_process(
    worker_id: int,
    df_chunk: pd.DataFrame,
    labeled_texts: set,
    result_queue: queue.Queue,
    progress_queue: queue.Queue
):
    """Worker function for parallel processing."""
    labeler = ESGLabeler(worker_id=worker_id)
    
    for idx, row in df_chunk.iterrows():
        result = label_single_row(row, labeler, labeled_texts)
        if result:
            result_queue.put(result)
        progress_queue.put(1)


def run_labeling(
    input_path: str,
    output_path: str,
    sample_size: int = None,
    resume: bool = True,
    workers: int = 1
):
    """
    Run LLM labeling pipeline.
    
    Args:
        input_path: Path to input CSV
        output_path: Path to output CSV  
        sample_size: Number of samples to process (None = all)
        resume: Resume from existing output if exists
        workers: Number of parallel workers (default: 1)
    """
    print(f"Loading data from: {input_path}")
    df = load_data(input_path, sample_size)
    print(f"Loaded {len(df)} texts" + (f" (sampled from larger dataset)" if sample_size else ""))
    
    # Resume from existing output if exists
    labeled_texts = set()
    existing_results = []
    
    if resume and os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        text_col = 'text' if 'text' in existing_df.columns else 'sentence'
        labeled_texts = set(existing_df[text_col].tolist())
        existing_results = existing_df.to_dict('records')
        print(f"Resuming from {len(labeled_texts)} already labeled texts")
    
    # Filter out already labeled
    df = df[~df['text'].isin(labeled_texts)]
    print(f"Texts to process: {len(df)}")
    
    if len(df) == 0:
        print("All texts already labeled!")
        return
    
    results = existing_results.copy()
    
    if workers <= 1:
        # Single-threaded processing
        labeler = ESGLabeler()
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Labeling"):
            result = label_single_row(row, labeler, labeled_texts)
            if result:
                results.append(result)
                
                # Save periodically
                if len(results) % 50 == 0:
                    pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
                    print(f"\nSaved {len(results)} results")
    else:
        # Multi-threaded processing
        print(f"Using {workers} parallel workers")
        
        # Split dataframe into chunks
        chunk_size = len(df) // workers
        chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(workers)]
        # Add remaining rows to last chunk
        if len(df) % workers != 0:
            chunks[-1] = pd.concat([chunks[-1], df.iloc[workers*chunk_size:]])
        
        # Queues for results and progress
        result_queue = queue.Queue()
        progress_queue = queue.Queue()
        
        # Start workers
        threads = []
        for i, chunk in enumerate(chunks):
            t = threading.Thread(
                target=worker_process,
                args=(i, chunk, labeled_texts, result_queue, progress_queue)
            )
            t.start()
            threads.append(t)
        
        # Progress bar
        total_to_process = len(df)
        pbar = tqdm(total=total_to_process, desc="Labeling")
        
        processed = 0
        while processed < total_to_process:
            try:
                progress_queue.get(timeout=0.1)
                processed += 1
                pbar.update(1)
                
                # Collect results from queue
                while not result_queue.empty():
                    results.append(result_queue.get_nowait())
                
                # Save periodically
                if len(results) % 100 == 0 and len(results) > len(existing_results):
                    pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
                    
            except queue.Empty:
                # Check if all threads are done
                if all(not t.is_alive() for t in threads):
                    break
        
        pbar.close()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Collect remaining results
        while not result_queue.empty():
            results.append(result_queue.get_nowait())
    
    # Final save
    pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nDone! Saved {len(results)} labeled texts to {output_path}")
    
    # Print statistics
    if results:
        result_df = pd.DataFrame(results)
        print("\n=== Label Distribution ===")
        print("\nESG Topics:")
        print(result_df['esg_label'].value_counts())
        print("\nSentence Types:")
        print(result_df['sentence_type'].value_counts())
        print("\nWashing Types:")
        print(result_df['washing_type'].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Labeling for EVINCE")
    parser.add_argument("--input", default="data/all_banks_sentences.csv", help="Input CSV path")
    parser.add_argument("--output", default="data/labeled_sentences.csv", help="Output CSV path")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of texts to label (default: all)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--no_resume", action="store_true", help="Start fresh, don't resume")
    
    args = parser.parse_args()
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_path = os.path.join(project_root, args.input)
    output_path = os.path.join(project_root, args.output)
    
    run_labeling(input_path, output_path, args.sample_size, resume=not args.no_resume, workers=args.workers)
