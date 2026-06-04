# ==========================================================
# Configuration Module
# ----------------------------------------------------------
# This file centralizes all system configuration parameters.
#
# It defines:
# - Input PDF file location
# - Paths to the final trained models
# - Output file destinations
# - Active model names for documentation/debugging
#
# The actual pipeline uses:
# - Fine-tuned HeBERT typed detector
# - Final SVM classifier
# - Regex detector
# - Decision Engine
# ==========================================================


# ----------------------------------------------------------
# Input PDF
# ----------------------------------------------------------

PDF_PATH = "input/hebrew_test_input_10_pages.pdf"


# ----------------------------------------------------------
# Final HeBERT typed leakage detector
# ----------------------------------------------------------

HEBERT_TYPED_MODEL_PATH = "models/hebert_typed_leak_detector"


# ----------------------------------------------------------
# Final SVM model
# ----------------------------------------------------------

SVM_MODEL_PATH = "models/svm_final/leak_classifier_svm_final.pkl"
SVM_VECTORIZER_PATH = "models/svm_final/tfidf_vectorizer_svm_final.pkl"


# ----------------------------------------------------------
# Active model names
# ----------------------------------------------------------

ACTIVE_HEBERT_MODEL_NAME = "fine_tuned_hebert_typed_leak_detector"
ACTIVE_SVM_MODEL_NAME = "svm_final"


# ----------------------------------------------------------
# Output files
# ----------------------------------------------------------

OUTPUT_JSON = "output/results.json"
OUTPUT_CSV = "output/results.csv"
OUTPUT_MARKED_PDF = "output/marked_leaks_report.pdf"