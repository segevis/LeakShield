# ==========================================================
# Machine Learning Classification Module
# ----------------------------------------------------------
# This module is responsible for detecting information leakage
# using a trained machine learning model.
#
# How it works:
# 1. Loads a pre-trained classifier (Logistic Regression / etc.)
# 2. Loads a TF-IDF vectorizer used during training
# 3. Converts input text into numerical features
# 4. Predicts:
#    - LEAK (sensitive information)
#    - NON_LEAK (non-sensitive information)
#
# Output:
# - Classification label
# - Confidence score
#
# This is the "decision layer" of the system.
# ==========================================================

import joblib

class LeakClassifier:
    def __init__(self, model_path, vectorizer_path, threshold=0.45):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.threshold = threshold

    def classify(self, text):
        vec = self.vectorizer.transform([text])
        probs = self.model.predict_proba(vec)[0]

        leak_prob = float(probs[1])

        return {
            "label": "LEAK" if leak_prob >= self.threshold else "NON_LEAK",
            "confidence": round(leak_prob, 4)
        }