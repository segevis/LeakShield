# ==========================================================
# Machine Learning Classifier Module
# ----------------------------------------------------------
# Loads a trained classifier and TF-IDF vectorizer.
#
# Supports:
# - Models with predict_proba, such as Logistic Regression
# - Linear SVM models with decision_function, such as LinearSVC
# ==========================================================

import math
import joblib


class LeakClassifier:
    def __init__(self, model_path, vectorizer_path, threshold=0.45):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.threshold = threshold

    def _score_with_predict_proba(self, X):
        probabilities = self.model.predict_proba(X)[0]

        if len(probabilities) == 1:
            return float(probabilities[0])

        return float(probabilities[1])

    def _score_with_decision_function(self, X):
        decision_score = self.model.decision_function(X)

        if hasattr(decision_score, "__len__"):
            decision_score = decision_score[0]

        # Convert SVM decision score to a pseudo-confidence in range [0, 1].
        # This is not a calibrated probability, but it provides a useful
        # confidence-like score for display and ranking.
        return 1 / (1 + math.exp(-float(decision_score)))

    def predict_leak_probability(self, text):
        X = self.vectorizer.transform([text])

        if hasattr(self.model, "predict_proba"):
            return self._score_with_predict_proba(X)

        if hasattr(self.model, "decision_function"):
            return self._score_with_decision_function(X)

        prediction = self.model.predict(X)[0]
        return float(prediction)

    def classify(self, text):
        leak_score = self.predict_leak_probability(text)

        if leak_score >= self.threshold:
            label = "LEAK"
        else:
            label = "NON_LEAK"

        return {
            "label": label,
            "confidence": round(leak_score, 4),
        }