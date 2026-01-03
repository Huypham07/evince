# EVINCE Data Directory

Place your data files here:

## Expected Files

- `all_banks_sentences.csv` - Raw sentences from bank annual reports
  - Columns: `sentence`, `bank`, `year`, `report_type`, `sentence_id`
  
- `labeled_sentences.csv` - LLM-labeled training data (generated)
  - Columns: `sentence`, `esg_label`, `sentence_type`, `washing_type`, `confidence`, etc.

## Data Format

### Input CSV (all_banks_sentences.csv)

```csv
sentence,bank,year,report_type,sentence_id
"Ngân hàng cam kết giảm phát thải carbon",BIDV,2023,annual,1
"Tổng tài sản đạt 2.1 triệu tỷ đồng",VCB,2023,annual,2
```

### Labeled CSV (labeled_sentences.csv)

```csv
sentence,esg_label,esg_confidence,sentence_type,washing_type,washing_confidence
"Cam kết giảm 15% CO2",Environmental_Performance,0.95,CLAIM,VAGUE_COMMITMENT,0.72
"Đã giảm 12% năm 2023",Environmental_Performance,0.97,EVIDENCE,NOT_WASHING,0.89
```

## Data Privacy

⚠️ This folder should not be committed to git if it contains sensitive data.
Add to `.gitignore` if needed.
