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
│   ├── raw_ocr_annual_report.zip # Raw text files
│   └── all_banks_sentences.csv   # Processed sentences
│
├── models/                 # 🧠 Classification models
├── claim_evidence/         # 🔗 Claim-Evidence Linking
├── training/               # 🏋️ Training pipeline
├── evaluation/             # 📈 Metrics
└── scripts/                # 📜 Utility scripts
    ├── llm_labeling.py     # LLM-based pseudo-labeling
    └── process_ocr.py      # OCR data processing
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

### 2. Prepare Data
Nếu bạn có file zip chứa các file text OCR (ví dụ: `data/raw_ocr_annual_report.zip`), chạy lệnh sau để chuẩn hóa dữ liệu:

```bash
python scripts/process_ocr.py --input data/raw_ocr_annual_report.zip --output data/all_banks_sentences.csv
```
Script sẽ tự động trích xuất Tên ngân hàng, Năm, và Loại báo cáo từ tên file và chia nhỏ thành các câu văn.

### 3. Generate Labels (Optional)
Nếu chưa có dữ liệu gán nhãn, sử dụng LLM để tạo nhãn tự động:

```bash
# Cấu hình Qwen/Gemini trong .env trước
python main.py label --input data/all_banks_sentences.csv --output data/labeled_data.csv --sample 2000
```

### 4. Train Model 🏋️
Bạn có thể train lại model trên dữ liệu của mình:

**Train ESG Topic Classifier:**
```bash
python main.py train \
    --model-type esg \
    --input data/labeled_data.csv \
    --epochs 5 \
    --output-dir ./checkpoints/esg
```

**Train Washing Detector:**
```bash
python main.py train \
    --model-type washing \
    --input data/labeled_data.csv \
    --epochs 10 \
    --output-dir ./checkpoints/washing
```

### 5. Document Analysis (Detection) 🔍
Phân tích tài liệu để tìm ESG-washing và **xem bằng chứng cụ thể**:

```bash
python main.py analyze --input data/all_banks_sentences.csv --bank BIDV --year 2023 --verbose
```

**Output mẫu:**
```
============================================================
DOCUMENT ANALYSIS RESULT
============================================================
Bank: BIDV | Year: 2023
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

# Single prediction
result = classifier.predict("Ngân hàng cam kết giảm phát thải carbon")
print(f"Label: {result.predicted_label}")
print(f"Confidence: {result.confidence:.2%}")

# Batch prediction
results = classifier.predict_batch(["Câu 1", "Câu 2", "Câu 3"])
```

### Document Analysis

```python
from evince.claim_evidence import DocumentAnalyzer

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
