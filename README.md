# LeakShield

LeakShield is a Python-based academic prototype for detecting sensitive-information leakage in Hebrew PDF documents.

The system extracts and normalizes text from PDF files, divides the text into generic structural units, detects structured patterns with Regex, analyzes context with a fine-tuned HeBERT Multi-Task model, and produces JSON, CSV, and marked PDF reports.

---

## Main Capabilities

- Hebrew PDF text extraction
- Text cleaning and normalization
- Generic dynamic segmentation based on document structure
- Regex detection for structured sensitive patterns
- HeBERT Multi-Task V2 semantic analysis
- Sequence-level `LEAK` / `NON_LEAK` classification
- Token-level sensitive-span detection
- Decision engine for combining detection evidence
- JSON and CSV result export
- Marked PDF report generation
- Desktop GUI
- Automated runtime, model, training, and integration tests

---

# System Architecture

```text
PDF document
    ↓
Text extraction
    ↓
Cleaning and normalization
    ↓
Dynamic structural segmentation
    ↓
Regex detection
    ↓
HeBERT Multi-Task V2
    ├── Sequence classification head
    └── Token classification head
    ↓
Decision engine
    ↓
JSON report + CSV report + marked PDF report
```

## Dynamic Text Segmentation

The segmentation mechanism is generic and does not contain document-specific headings, labels, company names, or sentences.

It uses structural signals such as:

- Blank lines and paragraph boundaries
- Natural line breaks
- Sentence-ending punctuation
- Bullets and numbered items
- Generic `label: value` structures
- Document-relative line statistics
- The active tokenizer limit when required

## HeBERT Multi-Task V2

The production model is based on a pretrained Hebrew HeBERT encoder that was fine-tuned for sensitive-information leakage detection.

The model contains two heads that share the same encoder:

- **Sequence classification head:** classifies the complete text unit as `LEAK` or `NON_LEAK`.
- **Token classification head:** identifies sensitive words or spans inside the text unit.

The sequence head produces `probability_leak`. The final prediction is determined by comparing this probability with the configured production threshold.

## Decision Engine

The decision engine combines:

- Regex evidence
- HeBERT sequence classification
- HeBERT token classification

It produces the final label and risk information for each analyzed text unit.

---

# Active Production Model

Production model directory:

```text
models/hebert_multitask_v2/best
```

Main model file:

```text
models/hebert_multitask_v2/best/model.safetensors
```

Expected SHA-256:

```text
A983C3230D983F990B7B801DB60AC8F20A1C623E6105F6907AC743ED22935CA1
```

The model file is stored in GitHub using Git LFS.

Production configuration:

```text
config/hebert_multitask_v2_production.json
```

Active sequence threshold:

```text
0.169112
```

---

# Requirements

- Windows 10 or Windows 11
- Python 3.13
- Git
- Git LFS
- Internet connection for the first dependency installation

---

# Download from GitHub

```powershell
git lfs install
git clone --branch runnable-production-package https://github.com/segevis/LeakShield.git
cd LeakShield
git lfs pull
```

Verify the model was downloaded:

```powershell
Get-Item .\models\hebert_multitask_v2\best\model.safetensors
```

The model size should be approximately 436 MB.

---

# First-Time Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

# Run the System

## Graphical Interface

Recommended for demonstrations:

```powershell
python gui.py
```

Then:

1. Click **Choose PDF**
2. Select a PDF
3. Click **Analyze PDF**
4. Wait for analysis to finish
5. Open the marked PDF, JSON, or CSV report

During analysis, the PDF-selection button is locked.

## Command Line

```powershell
python main.py
```

Default paths are configured in:

```text
config.py
```

---

# Output Files

```text
output/results.json
output/results.csv
output/marked_leaks_report.pdf
```

## `results.json`

Contains detailed structured information such as:

- Page number
- Text unit
- Final classification
- Risk information
- Regex detections
- Sequence-classification result
- Token-level detections
- Leakage probability
- Confidence
- Analysis status

## `results.csv`

Provides a table-friendly representation of the analysis results.

## `marked_leaks_report.pdf`

Visually highlights detected sensitive content in the original document.

---

# Project Structure

```text
LeakShield/
├── main.py
├── gui.py
├── analysis_pipeline.py
├── decision_engine.py
├── detectors.py
├── hebert_multitask_detector.py
├── sensitive_text_analyzer.py
├── pdf_processor.py
├── pdf_colored_report.py
├── utils.py
├── config.py
├── requirements.txt
├── README.md
├── config/
│   └── hebert_multitask_v2_production.json
├── data/
│   └── hebert_label_schema.json
├── input/
├── models/
│   └── hebert_multitask_v2/
│       └── best/
│           ├── model.safetensors
│           ├── config.json
│           ├── tokenizer.json
│           └── tokenizer_config.json
├── output/
├── tests/
└── training/
```

| File / Folder | Description |
|---|---|
| `main.py` | Command-line entry point |
| `gui.py` | Desktop interface |
| `analysis_pipeline.py` | End-to-end analysis pipeline |
| `decision_engine.py` | Combines Regex, sequence, and token evidence |
| `detectors.py` | Structured Regex detection |
| `hebert_multitask_detector.py` | Loads the tokenizer and production model |
| `sensitive_text_analyzer.py` | Public sensitive-text analysis interface |
| `pdf_processor.py` | PDF text extraction |
| `pdf_colored_report.py` | Marked PDF report generation |
| `utils.py` | Text normalization and generic dynamic segmentation |
| `config.py` | Runtime paths and outputs |
| `config/` | Production model configuration |
| `data/` | Label schema and model-related data |
| `models/` | Production and saved model artifacts |
| `training/` | Dataset, training, evaluation, calibration, and audit scripts |
| `tests/` | Unit, integration, runtime, dataset, report, and model tests |
| `input/` | Input and controlled demonstration PDFs |
| `output/` | Generated results |

---

# Training and Evaluation

Run commands from the project root with the virtual environment activated.

## Build Dataset

```powershell
python training\build_hebert_multitask_dataset.py
```

## Audit Dataset

```powershell
python training\audit_multitask_datasets.py
```

## Analyze Dataset Leakage Risk

```powershell
python training\analyze_multitask_leakage.py
```

## Train HeBERT Multi-Task

```powershell
python training\train_hebert_multitask.py
```

Updated V2 training:

```powershell
python training\train_hebert_multitask_v2.py
```

## Evaluate and Calibrate

```powershell
python training\evaluate_and_calibrate_hebert_multitask.py
```

Exact command-line arguments depend on the dataset and output paths. Review each script's argument parser before running a new experiment.

---

# Tests

## Production Runtime Tests

```powershell
python -m pytest tests\test_sensitive_text_analyzer.py tests\test_production_regex_hebert_contract.py -q
```

Expected current result:

```text
9 passed
```

## Full Test Suite

```powershell
python -m pytest tests -q
```

The suite includes tests for:

- Analysis pipeline
- Dataset construction
- Decision engine
- Evaluation and calibration
- Multi-task model
- Dataset auditing
- Data-leakage analysis
- Runtime integration
- Marked PDF report
- Regex and HeBERT production contract
- Sensitive-text analysis
- Training scripts

---

# Verify the Production Model

```powershell
Get-FileHash `
    .\models\hebert_multitask_v2\best\model.safetensors `
    -Algorithm SHA256
```

Expected:

```text
A983C3230D983F990B7B801DB60AC8F20A1C623E6105F6907AC743ED22935CA1
```

---

# Controlled Demonstration Scenarios

The project contains two controlled demonstration scenarios:

## Sensitive Business Document

Demonstrates positive leak detection and marked-PDF generation.

## Public Corporate Document

Demonstrates a valid public document with zero detected leaks.

These are controlled demonstration scenarios, not an independent blind benchmark.

---

# Important Notes and Limitations

- LeakShield is an academic engineering and research prototype.
- The model was fine-tuned from a pretrained Hebrew HeBERT model; it was not trained from zero.
- Much of the available labeled data was synthetic or constructed.
- Additional real Hebrew documents and expert human labeling are required before commercial deployment.
- False positives and false negatives can still occur on unseen domains and document structures.
- Performance depends on the quality of text extraction from the PDF.
- Scanned image-only PDFs require OCR before analysis.
- Raw extracted PDF text is not printed by default.
- The system currently runs locally.

---

# Academic Positioning

LeakShield demonstrates the technical feasibility of an end-to-end Hebrew PDF sensitive-information leakage detection system.

Its main contribution is the integration of:

- PDF processing
- Generic dynamic segmentation
- Structured Regex detection
- Fine-tuned Hebrew Transformer analysis
- Multi-task sequence and token classification
- Decision logic
- Visual and structured reporting
- GUI-based operation
- Automated validation

The main requirement for future production deployment is broader real-world labeled data and independent external evaluation.
