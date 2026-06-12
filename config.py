# ==========================================================
# LeakShield production configuration
# ----------------------------------------------------------
# Active detection architecture:
# - Regex for explicit structured patterns
# - HeBERT Multi-Task V2 with a shared encoder
#   - Token Head: localization and sensitive type
#   - Sequence Head: LEAK / NON_LEAK decision
#
# SVM and the old separate HeBERT models are not part of the
# active runtime pipeline.
# ==========================================================

PDF_PATH = "input/hebrew_test_input_10_pages.pdf"

OUTPUT_JSON = "output/results.json"
OUTPUT_CSV = "output/results.csv"
OUTPUT_MARKED_PDF = "output/marked_leaks_report.pdf"

HEBERT_MULTITASK_MODEL_PATH = "models/hebert_multitask_v2/best"
HEBERT_MULTITASK_CONFIG_PATH = "config/hebert_multitask_v2_production.json"
HEBERT_MULTITASK_TOKEN_THRESHOLD = 0.55
ACTIVE_HEBERT_MODEL_NAME = "hebert_multitask_v2_two_head"
ACTIVE_DETECTION_LAYERS = ("regex", "hebert_multitask_v2")
