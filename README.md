# EVINCE

<p align="center">
  <b>Evidence-Verified INtegrity Checker for ESG Claims</b><br>
  <i>A Novel Framework for ESG-Washing Detection in Vietnamese Banking Reports</i>
</p>

---

## 🎯 Overview

**EVINCE** là một framework Deep Learning để phát hiện **ESG-Washing** (tẩy xanh) trong báo cáo thường niên của các ngân hàng Việt Nam. Framework sử dụng **PhoBERT** làm encoder ngôn ngữ và tích hợp **Claim-Evidence Linking** để phân tích ở mức độ document.

### Tính năng chính

| Feature | Mô tả |
|---------|-------|
| 🏷️ **ESG Classification** | Phân loại câu văn vào 6 chủ đề ESG |
| 🔍 **Washing Detection** | Phát hiện 7 loại ESG-Washing với attention explainability |
| 📄 **Document Analysis** | Phân tích mức độ washing toàn bộ document |
| 🔗 **Claim-Evidence Linking** | Liên kết cam kết với bằng chứng hỗ trợ |
| 🤖 **LLM Labeling** | Tạo nhãn tự động với Qwen3 14B |

---

## 📦 Installation

### Prerequisites

```bash
Python >= 3.8
PyTorch >= 1.10
CUDA (optional, for GPU acceleration)
```

### Install Dependencies

```bash
pip install torch transformers pandas tqdm python-dotenv requests
```

### Clone Repository

```bash
git clone https://github.com/huypham71/EVINCE.git
cd EVINCE
```

---

## 🏗️ Project Structure

```
evince_v2/
├── main.py                 # 🚀 CLI entry point
├── README.md               # Documentation
├── .env.example            # Environment template
├── requirements.txt        # Dependencies
│
├── data/                   # 📊 Data directory
│   ├── all_banks_sentences.csv
│   └── labeled_sentences.csv
│
├── models/                 # 🧠 Classification models
│   ├── esg_topic_classifier.py    # ESG Topic Classifier (6 classes)
│   └── washing_detector.py        # Washing Detector (7 classes + attention)
│
├── claim_evidence/         # 🔗 Claim-Evidence Linking
│   ├── sentence_classifier.py     # CLAIM/EVIDENCE/CONTEXT classifier
│   ├── evidence_matcher.py        # Cross-encoder for claim-evidence scoring
│   ├── evidence_retriever.py      # Bi-encoder for evidence retrieval
│   └── document_analyzer.py       # Document-level analysis orchestrator
│
├── training/               # 🏋️ Training pipeline
│   ├── data_loader.py      # PyTorch datasets
│   └── train.py            # Training loop with checkpointing
│
├── evaluation/             # 📈 Metrics
│   └── metrics.py          # F1, Accuracy, ECE, Cohen's Kappa
│
├── scripts/                # 📜 Utility scripts
│   └── llm_labeling.py     # LLM-based pseudo-labeling
│
└── core/                   # ⚙️ Core utilities
    ├── config.py           # Configuration management
    └── qwen_client.py      # Qwen3 API client
```

---

## 🚀 Quick Start

### 1. Interactive Mode (Phân loại từng câu)

```bash
python main.py interactive
>>> Ngân hàng cam kết giảm phát thải carbon
→ Environmental_Performance (95.2%)
  Môi trường
```

### 2. Classify Single Text

```bash
python main.py classify --text "Ngân hàng đã giảm 15% lượng CO2 trong năm 2023"
```

**Output:**
```
Text: Ngân hàng đã giảm 15% lượng CO2 trong năm 2023
──────────────────────────────────────────────────
Label: Environmental_Performance
Label (VN): Môi trường
Confidence: 97.35%
Is ESG: True
```

### 3. Classify CSV File

```bash
python main.py classify --input data/sentences.csv --output results.csv
```

### 4. Analyze Document for Washing

```bash
python main.py analyze --input data/all_banks_sentences.csv --bank BIDV --year 2023
```

**Output:**
```
============================================================
DOCUMENT ANALYSIS RESULT
============================================================
Bank: BIDV
Year: 2023
Total Sentences: 1,234
────────────────────────────────────────────────────────────
Document Washing Index: 0.342
Total Claims: 156
Verified Claims: 89
High Risk Claims: 23
Average Verification Score: 0.571
============================================================
```

### 5. Generate Labels with LLM

```bash
# Configure .env first
cp .env.example .env
# Edit .env with your Qwen3 credentials

# Run labeling
python main.py label --input data/sentences.csv --output data/labeled.csv --sample 1000
```

---

## 📊 ESG Labels

### ESG Topic Classification (6 classes)

| Label | Description | Vietnamese |
|-------|-------------|------------|
| `Environmental_Performance` | Môi trường, khí hậu, năng lượng | Hiệu quả môi trường |
| `Social_Performance` | Nhân viên, cộng đồng, xã hội | Hiệu quả xã hội |
| `Governance_Performance` | Quản trị, đạo đức, tuân thủ | Hiệu quả quản trị |
| `ESG_Financing` | Tín dụng xanh, trái phiếu ESG | Tài chính ESG |
| `Strategy_and_Policy` | Chiến lược, chính sách ESG | Chiến lược & Chính sách |
| `Not_ESG_Related` | Không liên quan ESG | Không liên quan |

### Washing Types (7 classes)

| Type | Description |
|------|-------------|
| `NOT_WASHING` | Cam kết genuine, có bằng chứng rõ ràng |
| `VAGUE_COMMITMENT` | Cam kết mơ hồ, không có số liệu cụ thể |
| `SELECTIVE_DISCLOSURE` | Chỉ nêu điểm tốt, giấu điểm xấu |
| `SYMBOLIC_ACTION` | Hành động mang tính biểu tượng |
| `DECOUPLING` | Nói một đằng làm một nẻo |
| `MISLEADING_METRICS` | Số liệu gây hiểu lầm |
| `FUTURE_DEFLECTION` | Trì hoãn sang tương lai |

---

## 🐍 Python API

### ESG Classification

```python
from evince_v2.models import HuggingFaceESGClassifierInference

# Load pre-trained model from HuggingFace
classifier = HuggingFaceESGClassifierInference()

# Single prediction
result = classifier.predict("Ngân hàng cam kết giảm phát thải carbon")
print(f"Label: {result.predicted_label}")
print(f"Confidence: {result.confidence:.2%}")

# Batch prediction
results = classifier.predict_batch(["Câu 1", "Câu 2", "Câu 3"])
```

### Document Analysis

```python
from evince_v2.claim_evidence import DocumentAnalyzer

analyzer = DocumentAnalyzer(device="cuda")

result = analyzer.analyze_document(
    sentences=["Cam kết 1", "Bằng chứng 1", "Cam kết 2"],
    bank="BIDV",
    year=2023
)

print(f"Washing Index: {result.document_washing_index:.3f}")
print(f"High Risk Claims: {result.high_risk_claims}")
```

---

## 🔧 Configuration

### Environment Variables (.env)

```env
# Qwen3 LLM (for pseudo-labeling)
QWEN_BASE_URL=http://your-server:8000/v1/chat/completions
QWEN_AUTH_USERNAME=your_username
QWEN_AUTH_PASSWORD=your_password
QWEN_MODEL=Qwen3-14B

# Optional: Google Gemini
GOOGLE_API_KEY=your_api_key
```

---

## 📚 Pre-trained Models

| Model | HuggingFace Hub | Description |
|-------|-----------------|-------------|
| ESG Classifier | `huypham71/esgify_vn_class_weights` | 6-class ESG topic classifier |

---

## 📖 References

- **PhoBERT**: Nguyen & Tuan Nguyen (2020). PhoBERT: Pre-trained language models for Vietnamese.
- **ESGBERT**: Schimanski et al. (2024). ClimateBERT-based ESG classification.
- **A3CG Dataset**: Ong et al. (2025). Asian Anti-Greenwashing Claim-Context dataset.

---

## 👤 Author

**Huy Pham**  
University of Engineering and Technology (UET), Vietnam National University

---

## 📄 License

This project is for academic research purposes.
