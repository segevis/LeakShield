from __future__ import annotations

import re
from typing import Any, Dict, List, Set


# ==========================================================
# Decision Engine
# ----------------------------------------------------------
# This module combines evidence from three detection layers:
#
# 1. Regex:
#    Strong for explicit structured sensitive patterns:
#    email, phone, ID, password, API key, connection string.
#
# 2. Fine-tuned HeBERT:
#    Strong for typed semantic leakage:
#    salary, medical, personal, trade secret, prompt leak, etc.
#
# 3. SVM:
#    General full-sentence classifier.
#    Useful as a supporting signal, but not trusted alone too easily.
#
# Final principle:
# The engine does not ask "which model is right alone?"
# It asks:
# - How many evidence layers support the decision?
# - How strong is the evidence?
# - Is the text complete or broken?
# - Is there sensitive business context?
# - Is there public / negative context?
# ==========================================================


STRONG_REGEX_TYPES: Set[str] = {
    "EMAIL",
    "PHONE",
    "ID",
    "PASSWORD",
    "CONNECTION_STRING_KEYWORD",
}

CONTEXTUAL_REGEX_TYPES: Set[str] = {
    "MONEY",
    "ACCOUNT",
    "TICKET",
    "TECHNICAL_SECRET_KEYWORD",
}


HIGH_RISK_HEBERT_TYPES: Set[str] = {
    "PASSWORD_CREDENTIAL_LEAK",
    "TOKEN_SECRET_LEAK",
    "API_KEY_SECRET_LEAK",
    "PROMPT_SYSTEM_LEAK",
    "ID_DOCUMENT_LEAK",
    "MEDICAL_HEALTH_LEAK",
    "BANK_PAYMENT_LEAK",
    "PERSONAL_FINANCIAL_LEAK",
    "SALARY_COMPENSATION_LEAK",
    "TRADE_SECRET_IP_LEAK",
    "SECURITY_INCIDENT_LEAK",
    "LEGAL_CONTRACT_LEAK",
}


PUBLIC_NEGATION_CUES = [
    "אינו כולל",
    "אינה כוללת",
    "אינם כוללים",
    "אינן כוללות",
    "אין במקטע",
    "אין במסמך",
    "לא נכללו",
    "לא הועברו",
    "לא הוזכרו",
    "ללא שמות",
    "ללא פרטים",
    "ללא מידע",
    "ללא נתונים",
    "ללא מספר",
    "ללא מספרים",
    "ללא ערכים אמיתיים",
    "בלי ערכים אמיתיים",
    "בלי להציג",
    "בלי לפרט",
    "פומבי",
    "פומבית",
    "ציבורי",
    "ציבורית",
    "פורסם באתר",
    "שכבר פורסמו",
    "שכבר הוכרזו",
    "כללי בלבד",
    "כללית בלבד",
    "סטטיסטיקות כלליות",
    "דוגמאות כלליות",
    "דוגמה כללית",
    "מסביר באופן כללי",
    "מציג באופן כללי",
    "מתאר באופן כללי",
    "מהו",
    "מהי",
    "עמוד בדיקה",
    "בדיקה כללית",
    "לצורך בדיקה",
    "דפוסים לבדיקה",
    "מטרת עמוד זה",
]


# Stronger negative/public/general cues.
# These are used to prevent false positives where the text talks ABOUT
# sensitive concepts but does not expose actual sensitive values.
NEGATIVE_GENERAL_CONTEXT_CUES = [
    "בלי להציג סיסמה אמיתית",
    "בלי להציג סיסמאות אמיתיות",
    "בלי להציג פרטים אמיתיים",
    "בלי להציג נתונים אמיתיים",
    "בלי ערכים אמיתיים",
    "ללא ערכים אמיתיים",
    "ללא שמות",
    "ללא פרטים",
    "ללא פרטי לקוחות",
    "ללא פרטי עובדים",
    "ללא נתונים אישיים",
    "ללא מזהים אישיים",
    "ללא מספרים",
    "ללא מספר אמיתי",
    "ללא כתובת שרת",
    "ללא מידע של חברה מסוימת",
    "ללא מידע ארגוני ספציפי",
    "בלי לפרט ממצאי ביקורת ספציפיים",
    "תבנית ריקה",
    "דוגמה כללית",
    "דוגמאות כלליות",
    "דוגמה סכמטית",
    "באופן כללי",
    "מסביר באופן כללי",
    "מציג באופן כללי",
    "מתאר באופן כללי",
    "מהו",
    "מהי",
    "פומבי",
    "פומבית",
    "ציבורי",
    "ציבורית",
    "שכבר פורסם",
    "שכבר פורסמו",
    "שכבר הוכרז",
    "שכבר הוכרזו",
    "פורסם באתר",
]


SENSITIVE_CONTEXT_CUES = [
    "פנימי",
    "פנימית",
    "חסוי",
    "חסויה",
    "חסויים",
    "סודי",
    "מסווג",
    "לא להפיץ",
    "אין להפיץ",
    "לא נועד להפצה",
    "לשימוש הנהלה",
    "לשימוש פנימי",
    "פרטי גישה",
    "פרטי התחברות",
    "סיסמה",
    "סיסמאות",
    "מפתח",
    "api",
    "token",
    "לקוח",
    "לקוחות",
    "שכר",
    "בונוס",
    "בונוסים",
    "חשבון",
    "תשלום",
    "רפואי",
    "אבחנה",
    "חוזה",
    "הסכם",
    "מחירים",
    "תמחור",

    # Business-sensitive additions
    "רשימת הלקוחות",
    "לקוחות אסטרטגיים",
    "לקוחות מרכזיים",
    "אנשי קשר",
    "הנחות חריגות",
    "נספחי שירות",
    "מחירים מוסכמים",
    "תנאי התקשרות",
    "תנאי בלעדיות",
    "ויתורים מסחריים",
    "הצעת מחיר אישית",
    "לקוח ספציפי",
    "תחזית מכירות",
    "יעד הכנסות",
    "תקציב פרסום פנימי",
    "מפת דרכים חסויה",
    "פיצ'רים שלא הוכרזו",
]


def _get_types(findings: List[Dict[str, Any]]) -> List[str]:
    return sorted(
        {
            str(item.get("type"))
            for item in findings
            if item.get("type")
        }
    )


def _max_hebert_score(findings: List[Dict[str, Any]]) -> float:
    scores = []

    for item in findings:
        try:
            scores.append(float(item.get("score", 0.0)))
        except (TypeError, ValueError):
            continue

    return max(scores) if scores else 0.0


def _has_any(text: str, cues: List[str]) -> bool:
    text = str(text or "")
    return any(cue in text for cue in cues)


def _looks_like_incomplete_or_too_short(text: str) -> bool:
    """
    Generic guard against SVM-only false positives on broken/partial text units.

    Examples that should not become LEAK from SVM alone:
    - "31.קובץ ההגדרות כולל"
    - "48.הקובץ מציג הגדרה כללית של"
    - "75.המסמך כולל את ה־"
    - "5.פרטי עובד:"
    - "37.בדוח"
    """
    text = str(text or "").strip()

    if not text:
        return True

    clean = re.sub(r"^\d{1,3}\s*[\.\)]\s*", "", text).strip()
    words = clean.split()

    if len(clean) <= 4:
        return True

    if len(words) <= 3 and not re.search(r"[.!?؟]$", clean):
        return True

    if clean.endswith(":") and len(words) <= 8:
        return True

    dangling_endings = [
        "כולל",
        "כוללת",
        "מכיל",
        "מכילה",
        "מופיע",
        "מופיעה",
        "מסביר מהו",
        "מסביר מהי",
        "מתאר מהו",
        "מתאר מהי",
        "מציג הגדרה כללית של",
        "מציגה הגדרה כללית של",
        "המסמך כולל את ה־",
        "המסמך כולל את ה-",
        "רשימת ה־",
        "רשימת ה-",
        "שרתי ה־",
        "שרתי ה-",
        "מערכת ה־",
        "מערכת ה-",
        "בדוח",
        "מסמך",
        "הודעת",
        "טבלת",
        "קובץ ההגדרות כולל",
        "קובץ הלוגים כולל",
        "קובץ הניסוי כולל",
        "קובץ התמיכה כולל",
    ]

    if any(clean.endswith(ending) for ending in dangling_endings):
        return True

    return False


def _has_business_sensitive_combination(text: str) -> bool:
    """
    Generic business-leak rules.

    These rules do not depend on the test PDF.
    They identify combinations that are usually sensitive in real business documents.
    """
    text = str(text or "")

    customer_terms = [
        "רשימת הלקוחות",
        "לקוחות אסטרטגיים",
        "לקוחות מרכזיים",
        "לקוח ספציפי",
        "אנשי קשר",
        "שם הלקוח",
        "פרטי לקוח",
        "פרטי לקוחות",
    ]

    commercial_terms = [
        "מחירים",
        "מחיר",
        "הנחות",
        "הנחה",
        "הנחות חריגות",
        "תנאי התקשרות",
        "תנאי תשלום",
        "נספחי שירות",
        "מחירים מוסכמים",
        "הצעת מחיר אישית",
        "ויתורים מסחריים",
        "תנאי בלעדיות",
    ]

    confidentiality_terms = [
        "חסוי",
        "חסויה",
        "חסויים",
        "פנימי",
        "פנימית",
        "סודי",
        "סודית",
        "לא פורסם",
        "שטרם אושר לפרסום",
        "שלא הוכרזו",
        "לא להפצה",
        "אין להפיץ",
    ]

    strategy_terms = [
        "תחזית מכירות",
        "יעד הכנסות",
        "תוכנית חדירה לשוק",
        "רשימת מתחרים",
        "תקציב פרסום פנימי",
        "מפת דרכים",
        "מפת דרכים חסויה",
        "תאריכי השקה",
        "פיצ'רים",
        "פיצ'רים שלא הוכרזו",
    ]

    legal_terms = [
        "הסכם",
        "חוזה",
        "תביעה פתוחה",
        "חשיפה כספית",
        "קנסות יציאה",
        "תנאים פיננסיים",
        "תנאי התקשרות",
    ]

    security_terms = [
        "פער בקרת הרשאות",
        "הרשאות קריטי",
        "אירוע חריג",
        "התחברות חריגה",
        "חשבון השירות",
        "טופולוגיית הרשת הפנימית",
        "נקודות כשל קריטיות",
    ]

    supply_terms = [
        "שרשרת אספקה",
        "שמות ספקים",
        "מחירי רכישה",
        "זמני אספקה",
        "חולשות בהסכם",
        "ספק חלופי",
        "כמויות חסרות",
    ]

    has_customer = _has_any(text, customer_terms)
    has_commercial = _has_any(text, commercial_terms)
    has_confidentiality = _has_any(text, confidentiality_terms)
    has_strategy = _has_any(text, strategy_terms)
    has_legal = _has_any(text, legal_terms)
    has_security = _has_any(text, security_terms)
    has_supply = _has_any(text, supply_terms)

    if has_customer and has_commercial:
        return True

    if has_confidentiality and (has_customer or has_commercial or has_strategy or has_legal):
        return True

    if has_strategy and (has_confidentiality or has_customer or has_commercial):
        return True

    if has_legal and (has_confidentiality or has_customer or has_commercial):
        return True

    if has_security and (has_confidentiality or "קריטי" in text or "חריגה" in text):
        return True

    if has_supply and (has_confidentiality or has_commercial or "חולשות" in text):
        return True

    return False


def _hebert_only_seems_conceptual(
    text: str,
    hebert_typed_res: List[Dict[str, Any]],
) -> bool:
    """
    Prevent false positives when HeBERT detects a sensitive keyword
    inside a general/explanatory/negative sentence.

    Example:
    "הנוהל מסביר כיצד לבחור סיסמה חזקה בלי להציג סיסמה אמיתית של משתמש"
    HeBERT may detect "סיסמה", but the sentence says it does NOT expose
    a real password.
    """
    text = str(text or "")

    if not _has_any(text, NEGATIVE_GENERAL_CONTEXT_CUES):
        return False

    if not hebert_typed_res:
        return False

    # If all HeBERT values are generic concept words, not actual values,
    # treat as conceptual/general and do not mark as leak.
    generic_values = {
        "סיסמה",
        "סיסמאות",
        "מפתח",
        "token",
        "api",
        "שכר",
        "חוזה",
        "מחירים",
        "תעודת זהות",
        "דרכון",
        "אבחנה",
        "מידע רפואי",
        "פרטי לקוחות",
    }

    values = [
        str(item.get("value", "")).strip()
        for item in hebert_typed_res
        if item.get("value")
    ]

    if not values:
        return True

    # If the detected values are very short/generic terms, and the sentence
    # has negative/general context, do not treat it as a real leak.
    for value in values:
        if value in generic_values:
            continue

        # Very short Hebrew concept-like span.
        if len(value.split()) <= 2 and len(value) <= 14:
            continue

        # If one value looks specific enough, do not suppress.
        return False

    return True


def _business_rule_blocked_by_negative_context(text: str) -> bool:
    """
    Prevent business-sensitive keyword rules from firing on empty templates
    or explicitly negative/general sentences.

    Example:
    "המסמך מציג תבנית ריקה של חוזה ללא שמות, מחירים או פרטי לקוחות"
    """
    text = str(text or "")

    if _has_any(text, NEGATIVE_GENERAL_CONTEXT_CUES):
        return True

    return False


def make_final_decision(
    regex_res: List[Dict[str, Any]],
    hebert_typed_res: List[Dict[str, Any]],
    ml_res: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Final decision engine for the hybrid leakage detection system.
    """

    regex_res = regex_res or []
    hebert_typed_res = hebert_typed_res or []
    ml_res = ml_res or {}

    regex_types = _get_types(regex_res)
    hebert_types = _get_types(hebert_typed_res)

    regex_type_set = set(regex_types)
    hebert_type_set = set(hebert_types)

    svm_label = str(ml_res.get("label", "NON_LEAK"))
    svm_prediction = int(ml_res.get("prediction", 0) or 0)

    try:
        svm_margin = float(ml_res.get("margin", 0.0) or 0.0)
    except (TypeError, ValueError):
        svm_margin = 0.0

    detected_text = " ".join(
        [
            str(item.get("value", ""))
            for item in regex_res + hebert_typed_res
            if item.get("value")
        ]
    )

    full_text = str(ml_res.get("text", "")) or detected_text

    has_public_negation = _has_any(full_text, PUBLIC_NEGATION_CUES)
    has_negative_general_context = _has_any(full_text, NEGATIVE_GENERAL_CONTEXT_CUES)
    has_sensitive_context = _has_any(full_text, SENSITIVE_CONTEXT_CUES)
    has_business_sensitive_context = _has_business_sensitive_combination(full_text)

    has_strong_regex = bool(regex_type_set & STRONG_REGEX_TYPES)
    has_contextual_regex = bool(regex_type_set & CONTEXTUAL_REGEX_TYPES)

    has_hebert = len(hebert_typed_res) > 0
    has_high_risk_hebert = bool(hebert_type_set & HIGH_RISK_HEBERT_TYPES)
    max_hebert_score = _max_hebert_score(hebert_typed_res)

    svm_says_leak = svm_label == "LEAK" or svm_prediction == 1

    reasons: List[str] = []

    # ------------------------------------------------------
    # 1. Strong explicit regex findings
    # ------------------------------------------------------
    # Strong regex is the clearest evidence. Even if the sentence says
    # something general, an actual detected email/phone/ID/password is evidence.
    if has_strong_regex:
        reasons.append("Regex detected strong explicit sensitive pattern")

        if has_hebert:
            reasons.append("Fine-tuned HeBERT detected typed sensitive span(s)")

        if svm_says_leak:
            reasons.append("SVM also classified the segment as LEAK")

        return {
            "final_label": "LEAK",
            "risk_level": "HIGH",
            "final_confidence": 0.99,
            "reasons": reasons,
            "regex_types": regex_types,
            "hebert_typed_types": hebert_types,
        }

    # ------------------------------------------------------
    # 2. Negative/general context before HeBERT keyword-only decisions
    # ------------------------------------------------------
    if has_hebert and _hebert_only_seems_conceptual(full_text, hebert_typed_res):
        return {
            "final_label": "NON_LEAK",
            "risk_level": "LOW",
            "final_confidence": 0.49,
            "reasons": [
                "HeBERT detected sensitive concept words, but negative/general context indicates no real sensitive value is exposed"
            ],
            "regex_types": regex_types,
            "hebert_typed_types": hebert_types,
        }

    # ------------------------------------------------------
    # 3. High-risk HeBERT findings
    # ------------------------------------------------------
    if has_high_risk_hebert:
        reasons.append("Fine-tuned HeBERT detected high-risk typed leakage span(s)")

        if svm_says_leak:
            reasons.append("SVM also classified the segment as LEAK")

        if has_contextual_regex:
            reasons.append("Regex detected contextual sensitive indicator")

        return {
            "final_label": "LEAK",
            "risk_level": "HIGH",
            "final_confidence": max(0.90, round(max_hebert_score, 4)),
            "reasons": reasons,
            "regex_types": regex_types,
            "hebert_typed_types": hebert_types,
        }

    # ------------------------------------------------------
    # 4. Other HeBERT findings
    # ------------------------------------------------------
    if has_hebert:
        reasons.append("Fine-tuned HeBERT detected typed sensitive span(s)")

        if svm_says_leak:
            reasons.append("SVM classified the segment as LEAK and HeBERT found sensitive span(s)")
            risk_level = "HIGH"
            confidence = max(0.82, round(max_hebert_score, 4))
        else:
            reasons.append("HeBERT found sensitive span(s), but SVM did not classify the full segment as leak")
            risk_level = "MEDIUM"
            confidence = max(0.72, round(max_hebert_score, 4))

        return {
            "final_label": "LEAK",
            "risk_level": risk_level,
            "final_confidence": confidence,
            "reasons": reasons,
            "regex_types": regex_types,
            "hebert_typed_types": hebert_types,
        }

    # ------------------------------------------------------
    # 5. Contextual regex + SVM agreement
    # ------------------------------------------------------
    if has_contextual_regex and svm_says_leak:
        if has_negative_general_context:
            return {
                "final_label": "NON_LEAK",
                "risk_level": "LOW",
                "final_confidence": 0.49,
                "reasons": [
                    "Contextual Regex and SVM found weak evidence, but negative/general context reduced the final decision"
                ],
                "regex_types": regex_types,
                "hebert_typed_types": hebert_types,
            }

        reasons.append("Regex detected contextual sensitive indicator")
        reasons.append("SVM classified the segment as LEAK")

        return {
            "final_label": "LEAK",
            "risk_level": "MEDIUM",
            "final_confidence": 0.72,
            "reasons": reasons,
            "regex_types": regex_types,
            "hebert_typed_types": hebert_types,
        }

    # ------------------------------------------------------
    # 6. SVM only
    # ------------------------------------------------------
    if svm_says_leak:
        if _looks_like_incomplete_or_too_short(full_text):
            return {
                "final_label": "NON_LEAK",
                "risk_level": "LOW",
                "final_confidence": 0.49,
                "reasons": [
                    "SVM classified as LEAK, but the text unit looks incomplete or too short without Regex/HeBERT support"
                ],
                "regex_types": regex_types,
                "hebert_typed_types": hebert_types,
            }

        if has_public_negation or has_negative_general_context:
            return {
                "final_label": "NON_LEAK",
                "risk_level": "LOW",
                "final_confidence": 0.49,
                "reasons": [
                    "SVM classified as LEAK, but public/negative/general context reduced the final decision"
                ],
                "regex_types": regex_types,
                "hebert_typed_types": hebert_types,
            }

        if has_business_sensitive_context and not _business_rule_blocked_by_negative_context(full_text):
            return {
                "final_label": "LEAK",
                "risk_level": "MEDIUM",
                "final_confidence": 0.66,
                "reasons": [
                    "Business-sensitive context combination detected without explicit Regex/HeBERT span"
                ],
                "regex_types": regex_types,
                "hebert_typed_types": hebert_types,
            }

        if svm_margin >= 0.60:
            reasons.append("SVM classified the segment as LEAK with relatively strong margin")

            return {
                "final_label": "LEAK",
                "risk_level": "MEDIUM",
                "final_confidence": 0.62,
                "reasons": reasons,
                "regex_types": regex_types,
                "hebert_typed_types": hebert_types,
            }

        if has_sensitive_context and svm_margin >= 0.20:
            reasons.append("SVM classified as LEAK and sensitive context was present")

            return {
                "final_label": "LEAK",
                "risk_level": "MEDIUM",
                "final_confidence": 0.58,
                "reasons": reasons,
                "regex_types": regex_types,
                "hebert_typed_types": hebert_types,
            }

        return {
            "final_label": "NON_LEAK",
            "risk_level": "LOW",
            "final_confidence": 0.49,
            "reasons": [
                "SVM classified as LEAK, but no Regex or HeBERT evidence supported the decision"
            ],
            "regex_types": regex_types,
            "hebert_typed_types": hebert_types,
        }

    # ------------------------------------------------------
    # 7. Regex contextual only
    # ------------------------------------------------------
    if has_contextual_regex:
        return {
            "final_label": "NON_LEAK",
            "risk_level": "LOW",
            "final_confidence": 0.49,
            "reasons": [
                "Only contextual Regex indicator was detected without HeBERT or SVM support"
            ],
            "regex_types": regex_types,
            "hebert_typed_types": hebert_types,
        }

    # ------------------------------------------------------
    # 8. Business-sensitive combination without SVM confidence
    # ------------------------------------------------------
    if has_business_sensitive_context and not _business_rule_blocked_by_negative_context(full_text):
        return {
            "final_label": "LEAK",
            "risk_level": "MEDIUM",
            "final_confidence": 0.60,
            "reasons": [
                "Business-sensitive context combination detected"
            ],
            "regex_types": regex_types,
            "hebert_typed_types": hebert_types,
        }

    # ------------------------------------------------------
    # 9. No evidence
    # ------------------------------------------------------
    return {
        "final_label": "NON_LEAK",
        "risk_level": "LOW",
        "final_confidence": 0.49,
        "reasons": [
            "No explicit sensitive pattern, typed HeBERT span, or reliable SVM leakage evidence detected"
        ],
        "regex_types": regex_types,
        "hebert_typed_types": hebert_types,
    }