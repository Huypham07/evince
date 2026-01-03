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
| 🤖 **LLM Labeling** | Tạo nhãn tự động với Qwen3 14B |

---


## 🏗️ Project Structure

```
evince/
├── main.py                 # 🚀 CLI entry point
├── README.md               # Documentation
├── .env.example            # Environment template
├── metrics_visualizer.py   # Metrics plotting
│
├── data/                   # 📊 Data directory
│   ├── raw_ocr_annual_report.zip  # Raw OCR text files
│   └── semantic_chunks.csv        # Processed chunks
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
    ├── llm_labeling.py     # LLM-based pseudo-labeling
    └── process_ocr_semantic.py  # Smart OCR processing
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repo
git clone https://github.com/Huypham07/evince.git
cd evince

# Install dependencies
pip install torch transformers pandas tqdm python-dotenv requests scikit-learn

# Setup env
cp .env.example .env
```

### 2. Process Raw OCR Data → Semantic Chunks

Nếu bạn có file raw OCR (txt/zip), sử dụng **semantic chunking** để chia thành các đoạn có nghĩa:

```bash
# Xử lý file đơn
python main.py process --input data/bctn_2024_raw.txt --output data/chunks.csv

# Xử lý zip chứa nhiều file
python main.py process --input data/raw_ocr_annual_report.zip --output data/all_chunks.csv
```

**Output CSV sẽ có các cột:**
- `text`: Nội dung chunk (đảm bảo ≤500 tokens)
- `section`: Tên section (từ markdown headers `##`)
- `chunk_type`: `paragraph` hoặc `table`
- `bank`, `year`, `report_type`: Metadata từ filename
- `token_count`: Số token thực tế (đếm bằng PhoBERT tokenizer)

> 💡 **Tính năng**: Script sử dụng PhoBERT tokenizer để đếm token chính xác và tự động chia chunk nếu vượt 500 tokens.

### 3. Classify ESG Topics

```bash
# Classify từ file chunks
python main.py classify --input data/chunks.csv --output data/classified.csv

# Classify single text
python main.py classify --text "Ngân hàng cam kết giảm 30% phát thải carbon vào năm 2030"
```

### 4. Generate Labels with LLM (Optional)

Nếu chưa có dữ liệu gán nhãn, sử dụng LLM để tạo nhãn tự động:

```bash
# Cấu hình Qwen/Gemini trong .env trước
python main.py label --input data/chunks.csv --output data/labeled.csv --sample 2000
```

### 5. Train Model 🏋️

Bạn có thể train lại model trên dữ liệu của mình:

**Train ESG Topic Classifier:**
```bash
python main.py train \
    --model-type esg \
    --input data/labeled.csv \
    --epochs 5 \
    --output-dir ./checkpoints/esg
```

**Train Washing Detector:**
```bash
python main.py train \
    --model-type washing \
    --input data/labeled.csv \
    --epochs 10 \
    --output-dir ./checkpoints/washing
```

> 📝 **Note**: Models mặc định sử dụng `max_length=512` và `freeze_bert_layers=0` (full fine-tuning) để hiểu tốt ngữ cảnh đoạn văn.

### 6. Document Analysis (Detection) 🔍

Phân tích tài liệu để tìm ESG-washing và **xem bằng chứng cụ thể**:

```bash
python main.py analyze --input data/classified.csv --bank agribank --year 2024 --verbose
```

**Output mẫu:**
```
============================================================
DOCUMENT ANALYSIS RESULT
============================================================
Bank: agribank | Year: 2024
Document Washing Index: 0.412
High Risk Claims: 5
...
⚠️  HIGH RISK CLAIMS DETECTED (Washing Evidence):
────────────────────────────────────────────────────────────

[1] Claim: "Ngân hàng cam kết đạt Net Zero vào năm 2050"
    Risk Level: HIGH
    Verification Score: 0.120
    Evidence Found:
      (No relevant evidence found)

[2] Claim: "Chúng tôi luôn hỗ trợ cộng đồng bị ảnh hưởng thiên tai"
    Risk Level: MEDIUM
    Verification Score: 0.450
    Evidence Found:
      - [0.48] Ngân hàng đã quyên góp 5 tỷ đồng cho quỹ cứu trợ miền Trung.
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
from evince.models import HuggingFaceESGClassifierInference

# Load pre-trained model from HuggingFace
classifier = HuggingFaceESGClassifierInference()

# Single prediction (supports up to 512 tokens)
result = classifier.predict("Ngân hàng cam kết giảm phát thải carbon 30% vào năm 2030")
print(f"Label: {result.predicted_label}")
print(f"Confidence: {result.confidence:.2%}")

# Batch prediction
results = classifier.predict_batch(["Đoạn văn 1", "Đoạn văn 2", "Đoạn văn 3"])
```

### Document Analysis

```python
from evince.claim_evidence import DocumentAnalyzer

analyzer = DocumentAnalyzer(device="cuda")

result = analyzer.analyze_document(
    sentences=["Cam kết 1", "Bằng chứng 1", "Cam kết 2"],
    bank="agribank",
    year=2024
)

print(f"Washing Index: {result.document_washing_index:.3f}")
print(f"High Risk Claims: {result.high_risk_claims}")
```

### Process Raw OCR

```python
from evince.scripts.process_ocr_semantic import process_single_file, chunks_to_csv

# Process raw OCR file
chunks = process_single_file("data/bctn_2024_raw.txt")

# Save to CSV
chunks_to_csv(chunks, "data/chunks.csv")

# Each chunk has:
# - text (≤500 tokens)
# - section, chunk_type
# - bank, year, report_type
# - token_count
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

| Model | HuggingFace Hub | Max Tokens | Description |
|-------|-----------------|------------|-------------|
| ESG Classifier | `huypham71/esgify_vn_class_weights` | 512 | 6-class ESG topic classifier |

---

## 🔄 Complete Workflow

```bash
# 1. Process raw OCR → semantic chunks (with token limit)
python main.py process -i data/bctn_2024_raw.txt -o data/chunks.csv

# 2. Classify ESG topics
python main.py classify -i data/chunks.csv -o data/classified.csv

# 3. (Optional) Generate labels for training
python main.py label -i data/chunks.csv -o data/labeled.csv --sample 500

# 4. (Optional) Train custom model
python main.py train --model-type esg --input data/labeled.csv --epochs 5

# 5. Analyze for washing detection
python main.py analyze -i data/classified.csv --bank agribank --year 2024
```

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
