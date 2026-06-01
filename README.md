# LeakShield

LeakShield is a Python-based system for detecting sensitive information leakage in Hebrew PDF documents.

The system analyzes PDF files, extracts text, detects sensitive information using Regex, Hebrew NER, and a trained machine learning classifier, and exports the results to JSON and CSV files.

---

# How to Run the Project

## 1. Open Terminal in the Project Folder

Open PowerShell inside the project root folder:

```powershell
cd C:\Users\segev\Desktop\final_project
```

Make sure you are inside the project folder:

```powershell
dir
```

You should see files such as:

```text
main.py
gui.py
requirements.txt
config.py
training
models
data
input
output
```

---

## 2. Create a Virtual Environment

Run:

```powershell
python -m venv .venv
```

---

## 3. Activate the Virtual Environment

Run:

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, the terminal should show:

```text
(.venv)
```

Example:

```text
(.venv) PS C:\Users\segev\Desktop\final_project>
```

---

## 4. Install Required Libraries

Run:

```powershell
pip install -r requirements.txt
```

This installs all required Python libraries for the project.

---

## If PowerShell Blocks Activation

If you get an execution policy error, run this command once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate the virtual environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

# Run the System

There are two ways to run the system:

1. Run with GUI
2. Run from command line

---

## Option 1: Run with GUI

This is the recommended way to run and demonstrate the system.

```powershell
python gui.py
```

Then:

1. Click `Choose PDF`
2. Select a PDF file
3. Click `Analyze PDF`
4. Wait until the analysis is completed
5. Open the generated `results.json` or `results.csv` directly from the GUI

---

## Option 2: Run from Command Line

Run:

```powershell
python main.py
```

This analyzes the default PDF configured in:

```text
config.py
```

The default input PDF path is controlled by:

```python
PDF_PATH
```

The output files are controlled by:

```python
OUTPUT_JSON
OUTPUT_CSV
```

---

# Quick Copy-Paste Commands

## First-Time Setup

Copy and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run GUI

```powershell
python gui.py
```

## Run Command-Line Version

```powershell
python main.py
```

---

# Output Files

After running the analysis, the system creates:

```text
output/results.json
output/results.csv
```

## `results.json`

Detailed structured output that includes:

* Page number
* Sentence / text unit
* Regex detections
* HeBERT NER detections
* ML classification
* Confidence score

## `results.csv`

Table-friendly output that includes:

* Page number
* Sentence
* Classification label
* Confidence
* Regex types
* Regex values

---

# Project Structure

```text
final_project/
├── main.py
├── gui.py
├── analysis_pipeline.py
├── config.py
├── pdf_processor.py
├── utils.py
├── detectors.py
├── ml_classifier.py
├── requirements.txt
├── data/
├── input/
├── models/
├── output/
└── training/
```

| File / Folder          | Description                                              |
| ---------------------- | -------------------------------------------------------- |
| `gui.py`               | Desktop GUI for selecting and analyzing PDF files        |
| `main.py`              | Command-line entry point                                 |
| `analysis_pipeline.py` | Main reusable analysis pipeline                          |
| `config.py`            | Paths for input, output, model and vectorizer            |
| `pdf_processor.py`     | PDF text extraction                                      |
| `utils.py`             | Text normalization and segmentation                      |
| `detectors.py`         | Regex and HeBERT NER detection                           |
| `ml_classifier.py`     | Loads the trained classifier and performs classification |
| `training/`            | Training and evaluation scripts                          |
| `data/`                | Training datasets                                        |
| `models/`              | Saved trained models and vectorizers                     |
| `output/`              | Generated result files                                   |

---

# Final Selected Model

The final model used by the system is:

```text
19k augmented dataset + TF-IDF + Linear SVM
```

Model files:

```text
models/comparison_svm_19000.pkl
models/comparison_tfidf_svm_19000.pkl
```

These paths are configured in:

```text
config.py
```

---

# Training and Evaluation

All training commands should be executed from the project root folder.

## Train Original Logistic Regression Model

```powershell
python training\train_model.py
```

## Run Stratified K-Fold Evaluation

```powershell
python training\train_model_stratified_kfold.py
```

## Run Grouped Split Evaluation

```powershell
python training\train_model_grouped_split.py
```

## Run GroupKFold on 18k Dataset

```powershell
python training\train_model_group_kfold.py
```

## Train and Evaluate SVM on 18k Dataset

```powershell
python training\train_model_svm_group_kfold.py
```

## Train and Evaluate Logistic Regression on 19k Dataset

```powershell
python training\train_model_group_kfold_19000.py
```

## Train and Evaluate SVM on 19k Dataset

```powershell
python training\train_model_svm_group_kfold_19000.py
```

## Run Full Model Comparison

```powershell
python training\run_all_model_comparisons.py
```

This creates:

```text
output/model_comparison_summary.csv
output/model_comparison_folds.csv
```

---

# Full Training / Evaluation Block

To run the main evaluation scripts one after another, copy this block:

```powershell
python training\train_model.py
python training\train_model_stratified_kfold.py
python training\train_model_grouped_split.py
python training\train_model_group_kfold.py
python training\train_model_svm_group_kfold.py
python training\train_model_group_kfold_19000.py
python training\train_model_svm_group_kfold_19000.py
python training\run_all_model_comparisons.py
```

---

# Final Model Comparison

| Experiment                 | Dataset       | Model               | Accuracy | Precision | Recall |     F1 |
| -------------------------- | ------------- | ------------------- | -------: | --------: | -----: | -----: |
| `svm_19000_augmented`      | 19k augmented | SVM                 |   0.9868 |    0.9788 | 1.0000 | 0.9888 |
| `logistic_19000_augmented` | 19k augmented | Logistic Regression |   0.9819 |    0.9788 | 0.9908 | 0.9841 |
| `svm_18000`                | 18k combined  | SVM                 |   0.9583 |    0.9551 | 0.9760 | 0.9634 |
| `logistic_18000`           | 18k combined  | Logistic Regression |   0.9532 |    0.9542 | 0.9672 | 0.9590 |

The selected final model is:

```text
svm_19000_augmented
```

---

# Notes

* The system is a local academic prototype.
* The datasets are synthetic / semi-synthetic.
* Raw PDF text logging is disabled by default to avoid exposing sensitive content.
* The GUI is intended for local demonstration and testing.
* Scanned image-based PDFs are not currently supported unless text can be extracted from them.
