"""
EVINCE: LLM Labeling Script

Generate training labels using Qwen3 14B for:
1. ESG Topic Classification (E, S, G, Financing, Policy, Non-ESG)
2. Sentence Type Classification (CLAIM, EVIDENCE, CONTEXT, NON-ESG)
3. Washing Detection (7 washing types)

Usage:
    python -m evince.scripts.llm_labeling --sample_size 1000 --output labeled_data.csv
"""

import os
import sys
import json
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import QwenClient, Config


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


class ESGLabeler:
    """LLM-based labeler for ESG data."""
    
    def __init__(self):
        self.client = QwenClient()
        print(f"Initialized QwenClient with model: {Config.QWEN_MODEL}")
    
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
    """Load sentences from CSV."""
    df = pd.read_csv(csv_path)
    
    # Filter short/noisy sentences
    df = df[df['sentence'].str.len() > 20]
    df = df[df['sentence'].str.split().str.len() > 5]
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    return df.reset_index(drop=True)


def run_labeling(
    input_path: str,
    output_path: str,
    sample_size: int = 1000,
    resume: bool = True
):
    """Run LLM labeling pipeline."""
    print(f"Loading data from: {input_path}")
    df = load_data(input_path, sample_size)
    print(f"Loaded {len(df)} sentences")
    
    # Resume from existing output if exists
    labeled_sentences = set()
    existing_results = []
    
    if resume and os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        labeled_sentences = set(existing_df['sentence'].tolist())
        existing_results = existing_df.to_dict('records')
        print(f"Resuming from {len(labeled_sentences)} already labeled sentences")
    
    # Initialize labeler
    labeler = ESGLabeler()
    
    # Label sentences
    results = existing_results.copy()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Labeling"):
        sentence = row['sentence']
        
        if sentence in labeled_sentences:
            continue
        
        try:
            result = labeler.label_full(sentence)
            result.update({
                "bank": row.get('bank', ''),
                "year": row.get('year', 0),
                "report_type": row.get('report_type', ''),
            })
            results.append(result)
            
            # Save periodically
            if len(results) % 50 == 0:
                pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"Saved {len(results)} results")
                
        except Exception as e:
            print(f"Error labeling: {e}")
            continue
    
    # Final save
    pd.DataFrame(results).to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Done! Saved {len(results)} labeled sentences to {output_path}")
    
    # Print statistics
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
    parser.add_argument("--sample_size", type=int, default=1000, help="Number of sentences to label")
    parser.add_argument("--no_resume", action="store_true", help="Start fresh, don't resume")
    
    args = parser.parse_args()
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_path = os.path.join(project_root, args.input)
    output_path = os.path.join(project_root, args.output)
    
    run_labeling(input_path, output_path, args.sample_size, resume=not args.no_resume)
