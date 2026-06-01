# ==========================================================
# Configuration Module
# ----------------------------------------------------------
# This file centralizes all system configuration parameters.
# It defines:
# - Input PDF file location
# - Paths to trained ML models
# - Output file destinations (JSON and CSV)
#
# This allows easy modification of paths without changing
# the core logic of the system.
# ==========================================================

PDF_PATH = "input\hebrew_test_input_10_pages.pdf"

MODEL_PATH = "models/leak_classifier.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

OUTPUT_JSON = "output/results.json"
OUTPUT_CSV = "output/results.csv"