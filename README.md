# evince

<p align="center">
  <b>Evidence-Verified INtegrity Checker for ESG Claims</b><br>
  <i>A Novel Framework for ESG-Washing Detection in Vietnamese Banking Reports</i>
</p>

---

## 🎯 Overview

**evince** là một framework Deep Learning để phát hiện **ESG-Washing** (tẩy xanh) trong báo cáo thường niên của các ngân hàng Việt Nam. Framework sử dụng **PhoBERT** làm encoder ngôn ngữ và tích hợp **Claim-Evidence Linking** để phân tích ở mức độ document.

### Tính năng chính

| Feature | Mô tả |
|---------|-------|
| 🏷️ **ESG Classification** | Phân loại câu văn vào 6 chủ đề ESG (max 512 tokens) |
| 🔍 **Washing Detection** | Phát hiện 7 loại ESG-Washing với attention explainability |
| 📄 **Document Analysis** | Phân tích mức độ washing toàn bộ document |
| 🔗 **Claim-Evidence Linking** | Liên kết cam kết với bằng chứng hỗ trợ |
| 📝 **Semantic Chunking** | Xử lý raw OCR thành semantic chunks với token limit |
| 🤖 **LLM Labeling** | Tạo nhãn tự động với Gemini, Bedrock, hoặc Qwen3 |
| 🏋️ **Training Pipeline** | Train custom models với labeled data |

---

## 🏗️ Project Structure

```
evince/
├── main.py                 # 🚀 CLI entry point
├── README.md               # Documentation
├── .env.example            # Environment template
├── requirements.txt        # Dependencies
│
├── data/                   # 📊 Data directory
│   ├── raw_ocr_annual_report.zip  # Raw OCR text files
│   ├── all_chunks.csv             # Processed chunks
│   └── labeled.csv                # LLM-labeled data
│
├── core/                   # 🔧 Core utilities
│   ├── config.py           # Configuration
│   ├── gemini_client.py    # Google Gemini LLM
│   ├── bedrock_client.py   # AWS Bedrock LLM
│   └── qwen_client.py      # Qwen3 LLM
│
├── models/                 # 🧠 Classification models
│   ├── esg_topic_classifier.py    # ESG 6-class classifier (512 tokens)
│   └── washing_detector.py        # Washing 7-class detector
├── claim_evidence/         # 🔗 Claim-Evidence Linking
├── training/               # 🏋️ Training pipeline
│   ├── train.py            # Trainer class
│   └── data_loader.py      # Dataset & DataLoader (512 tokens)
├── evaluation/             # 📈 Metrics
└── scripts/                # 📜 Utility scripts
    ├── llm_labeling.py     # LLM-based pseudo-labeling (multi-threaded)
    └── process_ocr_semantic.py  # Smart OCR processing
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repo
git clone https://github.com/Huypham07/evince.git
cd evince

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Setup env
cp .env.example .env
# Edit .env với API keys của bạn
```

### 2. Process Raw OCR → Semantic Chunks

Xử lý raw OCR files thành semantic chunks với token limit:

```bash
# Xử lý file đơn
python main.py process --input data/bctn_2024_raw.txt --output data/chunks.csv

# Xử lý zip chứa nhiều file (ví dụ: 11 banks × 5 years)
python main.py process --input data/raw_ocr_annual_report.zip --output data/all_chunks.csv
```

**Output:**
```
📊 Statistics:
  Total chunks: 30604
  Paragraph chunks: 18266
  Table chunks: 12338
  Average token count: 160
  Banks: ['vib', 'viettinbank', 'mbbank', 'shb', 'bsc', 'vietcombank', ...]
  Years: [2015, 2017, 2018, 2020, 2021, 2022, 2023, 2024]
```

### 3. Generate Labels with LLM

Tạo training labels với LLM (Gemini/Bedrock/Qwen):

```bash
# Cấu hình trong .env:
# LLM_PROVIDER=gemini
# GOOGLE_API_KEY=your_api_key

# Label toàn bộ dataset (multi-threaded)
python main.py label -i data/all_chunks.csv -o data/labeled.csv --workers 4

# Hoặc sample nhỏ để test
python main.py label -i data/all_chunks.csv -o data/labeled.csv --sample 100
```

**Output:**
```
=== Label Distribution ===

ESG Topics:
  Non-ESG: 18836 (61.7%)
  G: 7588 (24.8%)
  S: 2726 (8.9%)
  Financing: 688 (2.3%)
  E: 399 (1.3%)
  Policy: 310 (1.0%)

Washing Types:
  NOT_WASHING: 26452 (86.6%)
  VAGUE_COMMITMENT: 2383 (7.8%)
  SYMBOLIC_ACTION: 787 (2.6%)
  FUTURE_DEFLECTION: 530 (1.7%)
```

### 4. Train Custom Models 🏋️

**Train ESG Topic Classifier:**
```bash
python main.py train \
    --model-type esg \
    --input data/labeled.csv \
    --epochs 5 \
    --batch-size 16 \
    --output-dir ./checkpoints/esg
```

**Train Washing Detector:**
```bash
python main.py train \
    --model-type washing \
    --input data/labeled.csv \
    --epochs 10 \
    --batch-size 16 \
    --output-dir ./checkpoints/washing
```

**Training Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--model-type` | required | `esg` or `washing` |
| `--epochs` | 5 | Number of epochs |
| `--batch-size` | 16 | Batch size |
| `--learning-rate` | 2e-5 | Learning rate |
| `--max-length` | 512 | Max token length |
| `--val-split` | 0.1 | Validation split |
| `--freeze-layers` | 0 | BERT layers to freeze |
| `--device` | auto | cpu/cuda/auto |

### 5. Classify ESG Topics

```bash
# Classify từ file
python main.py classify --input data/chunks.csv --output data/classified.csv

# Classify single text
python main.py classify --text "Ngân hàng cam kết giảm 30% phát thải carbon vào năm 2030"
```

### 6. Document Analysis (Washing Detection) 🔍

```bash
python main.py analyze --input data/classified.csv --bank agribank --year 2024
```

**Output:**
```
============================================================
DOCUMENT ANALYSIS RESULT
============================================================
Bank: agribank | Year: 2024
Document Washing Index: 0.412
High Risk Claims: 5

⚠️  HIGH RISK CLAIMS DETECTED:
────────────────────────────────────────────────────────────
[1] Claim: "Ngân hàng cam kết đạt Net Zero vào năm 2050"
    Risk Level: HIGH
    Verification Score: 0.120
    Evidence Found: (No relevant evidence found)
```

---

## 📊 Label Definitions

### ESG Topic Classification (6 classes)

| Label | Code | Description |
|-------|------|-------------|
| Environmental | `E` | Môi trường, khí hậu, năng lượng, carbon |
| Social | `S` | Nhân viên, cộng đồng, sức khỏe, đào tạo |
| Governance | `G` | Quản trị, đạo đức, tuân thủ, rủi ro |
| ESG Financing | `Financing` | Tín dụng xanh, trái phiếu ESG |
| Policy | `Policy` | Chiến lược, chính sách ESG |
| Non-ESG | `Non-ESG` | Không liên quan ESG |

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

## 🔧 Configuration

### Environment Variables (.env)

```env
# ==============================================================================
# LLM PROVIDER SELECTION
# Options: "gemini", "bedrock", "qwen"
# ==============================================================================
LLM_PROVIDER=gemini

# ==============================================================================
# GOOGLE GEMINI (recommended)
# ==============================================================================
GOOGLE_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash-lite

# ==============================================================================
# AWS BEDROCK (alternative)
# ==============================================================================
AWS_BEDROCK_REGION=us-east-1
BEDROCK_MODEL=claude-3.7-sonnet
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# ==============================================================================
# QWEN3 (self-hosted)
# ==============================================================================
QWEN_BASE_URL=http://your-server:8000/v1/chat/completions
QWEN_AUTH_USERNAME=username
QWEN_AUTH_PASSWORD=password
```

---

## 🔄 Complete Workflow

```bash
# 1. Process raw OCR → semantic chunks
python main.py process -i data/raw_ocr_annual_report.zip -o data/all_chunks.csv

# 2. Generate labels with LLM (multi-threaded)
python main.py label -i data/all_chunks.csv -o data/labeled.csv -w 4

# 3. Train ESG classifier
python main.py train --model-type esg --input data/labeled.csv --epochs 5

# 4. Train Washing detector
python main.py train --model-type washing --input data/labeled.csv --epochs 10

# 5. Classify new documents
python main.py classify -i new_data.csv -o classified.csv

# 6. Analyze for washing
python main.py analyze -i classified.csv --bank bidv --year 2024
```

---

## 🐍 Python API

### ESG Classification

```python
from models import HuggingFaceESGClassifierInference

# Load pre-trained model
classifier = HuggingFaceESGClassifierInference()

# Predict
result = classifier.predict("Ngân hàng cam kết giảm phát thải carbon")
print(f"Label: {result.predicted_label}, Confidence: {result.confidence:.2%}")

# Batch prediction
results = classifier.predict_batch(["Text 1", "Text 2", "Text 3"])
```

### Document Analysis

```python
from claim_evidence import DocumentAnalyzer

analyzer = DocumentAnalyzer(device="cuda")
result = analyzer.analyze_document(sentences, bank="agribank", year=2024)

print(f"Washing Index: {result.document_washing_index:.3f}")
print(f"High Risk Claims: {result.high_risk_claims}")
```

### LLM Clients

```python
from core import GeminiClient, BedrockClient

# Gemini
client = GeminiClient()
result = client.generate_content("Classify this text...")

# AWS Bedrock
client = BedrockClient(region="us-east-1", model_id="claude-3.7-sonnet")
result = client.generate_content("Classify this text...")
```

---

## 📚 Pre-trained Models

| Model | HuggingFace Hub | Description |
|-------|-----------------|-------------|
| ESG Classifier | `huypham71/esgify_vn_class_weights` | 6-class ESG topic classifier |

---

## 📖 References

- **PhoBERT**: Nguyen & Tuan Nguyen (2020). Pre-trained language models for Vietnamese.
- **ESGBERT**: Schimanski et al. (2024). ClimateBERT-based ESG classification.
- **A3CG Dataset**: Ong et al. (2025). Asian Anti-Greenwashing Claim-Context dataset.

---

## 👤 Author

**Huy Pham**  
University of Engineering and Technology (UET), Vietnam National University

---

## 📄 License

This project is for academic research purposes.
