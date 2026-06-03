from __future__ import annotations

import os
import pickle
from typing import Any, Dict


try:
    from config import SVM_MODEL_PATH, SVM_VECTORIZER_PATH
except ImportError:
    SVM_MODEL_PATH = "models/svm_final/leak_classifier_svm_final.pkl"
    SVM_VECTORIZER_PATH = "models/svm_final/tfidf_vectorizer_svm_final.pkl"


MODEL_PATH = os.getenv("SVM_FINAL_MODEL_PATH", SVM_MODEL_PATH)
VECTORIZER_PATH = os.getenv("SVM_FINAL_VECTORIZER_PATH", SVM_VECTORIZER_PATH)


_model = None
_vectorizer = None


def _load_model_and_vectorizer():
    global _model, _vectorizer

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"SVM final model not found: {MODEL_PATH}")

        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)

    if _vectorizer is None:
        if not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(f"SVM final vectorizer not found: {VECTORIZER_PATH}")

        with open(VECTORIZER_PATH, "rb") as f:
            _vectorizer = pickle.load(f)

    return _model, _vectorizer


def predict_svm_final(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return {
            "label": "NON_LEAK",
            "prediction": 0,
            "confidence": 0.51,
            "margin": 0.0,
            "source": "svm_final_38k_dedup_balanced_word_c_0_5",
        }

    model, vectorizer = _load_model_and_vectorizer()

    x = vectorizer.transform([text])
    pred = int(model.predict(x)[0])

    # LinearSVC gives margin, not probability.
    margin = float(model.decision_function(x)[0])
    confidence = min(0.99, max(0.51, abs(margin) / 3.0))

    return {
        "label": "LEAK" if pred == 1 else "NON_LEAK",
        "prediction": pred,
        "confidence": round(confidence, 4),
        "margin": round(margin, 4),
        "source": "svm_final_38k_dedup_balanced_word_c_0_5",
    }


def main():
    examples = [
        "המסמך כולל api_key של שירות התשלומים.",
        "המסמך מסביר באופן כללי מהו api_key.",
        "הדוח כולל את רשימת הלקוחות האסטרטגיים של החברה.",
        "הדוח מסביר באופן כללי כיצד מנהלים קשרי לקוחות.",
        "המסמך חושף את אסטרטגיית התמחור של המוצר החדש.",
        "המצגת מסבירה באופן כללי מהי אסטרטגיית תמחור.",
        "הקובץ כולל כתובת אימייל פרטית של לקוח.",
        "המסמך מסביר מהי כתובת אימייל.",
    ]

    for example in examples:
        print("=" * 80)
        print(example)
        print(predict_svm_final(example))


if __name__ == "__main__":
    main()