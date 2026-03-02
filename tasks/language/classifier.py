"""
tasks/language/classifier.py
------------------------------
Text classification pipeline for competition NLP tasks.

Covers the two patterns seen in AINM/DM-i-AI:
  1. Binary classification      – Is a statement true or false?
  2. Multi-class categorization – Assign one of N medical/domain labels?

Tiers (automatically selected based on available dependencies):
  - Tier 1 (best):   HuggingFace transformers pipeline (zero-shot or fine-tuned)
  - Tier 2 (fast):   Sentence embeddings + sklearn classifier (competition sweet spot)
  - Tier 3 (safe):   TF-IDF + sklearn (always works, strong baseline for short texts)

Usage:
    from tasks.language.classifier import TextClassifier

    # Tier 2 — embeddings + logistic regression
    clf = TextClassifier(strategy="embeddings")
    clf.fit(texts_train, labels_train)
    preds = clf.predict(texts_test)
    clf.save("models/text_clf_v1.pkl")

    # Zero-shot (no training data needed)
    clf = TextClassifier(strategy="zero_shot", candidate_labels=["urgent", "non-urgent"])
    preds = clf.predict(texts_test)
"""

import os
import warnings
import time
from typing import Literal, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

_HAS_SENTENCE_TRANSFORMERS = False
_HAS_TRANSFORMERS = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    pass

try:
    from transformers import pipeline as hf_pipeline
    _HAS_TRANSFORMERS = True
except ImportError:
    pass

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")


class TextClassifier:
    """
    Flexible text classifier.

    Strategies:
        "embeddings"   → Sentence embeddings + LogisticRegression (recommended)
        "tfidf"        → TF-IDF + LogisticRegression/SVM (fallback)
        "zero_shot"    → HuggingFace zero-shot classification (no training data needed)
        "few_shot_llm" → Uses the factory LLM for few-shot prompting via LangChain

    For most competition tasks, "embeddings" gives the best accuracy/speed tradeoff.
    "zero_shot" is the fastest path when you have 0 labeled training examples.
    """

    def __init__(
        self,
        strategy: Literal["embeddings", "tfidf", "zero_shot", "few_shot_llm"] = "embeddings",
        candidate_labels: Optional[list[str]] = None,
        embedding_model: str = HF_EMBEDDING_MODEL,
        zero_shot_model: str = "facebook/bart-large-mnli",
    ):
        self.strategy = strategy
        self.candidate_labels = candidate_labels or []
        self.embedding_model_name = embedding_model
        self.zero_shot_model_name = zero_shot_model

        self._embedding_model: Optional[SentenceTransformer] = None
        self._classifier: Optional[Pipeline] = None
        self._label_encoder = LabelEncoder()
        self._zero_shot_pipe = None
        self._fitted = False

        if strategy == "embeddings" and not _HAS_SENTENCE_TRANSFORMERS:
            warnings.warn("sentence-transformers not installed. Falling back to 'tfidf'.", stacklevel=2)
            self.strategy = "tfidf"

        if strategy == "zero_shot" and not _HAS_TRANSFORMERS:
            warnings.warn("transformers not installed. Falling back to 'tfidf'.", stacklevel=2)
            self.strategy = "tfidf"

    def fit(
        self,
        texts: list[str],
        labels: list[str],
        classifier_type: Literal["logistic", "svm"] = "logistic",
    ) -> "TextClassifier":
        """
        Fit the classifier on training data.

        Args:
            texts:           List of raw text strings
            labels:          List of string labels (will be label-encoded internally)
            classifier_type: Sklearn classifier to place on top of features
        """
        if self.strategy == "zero_shot":
            print("ℹ️  Zero-shot strategy requires no training data. Skipping fit().")
            self._setup_zero_shot()
            return self

        if self.strategy == "few_shot_llm":
            # Store examples for in-context learning (no actual training)
            self._few_shot_examples = list(zip(texts[:10], labels[:10]))
            self._fitted = True
            return self

        print(f"🔤 Fitting TextClassifier — strategy={self.strategy}, n_samples={len(texts)}")
        t0 = time.time()

        encoded_labels = self._label_encoder.fit_transform(labels)
        features = self._extract_features(texts, fit=True)

        sk_model = (
            LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, C=1.0)
            if classifier_type == "logistic"
            else LinearSVC(max_iter=3000, random_state=RANDOM_SEED)
        )
        self._classifier = sk_model
        self._classifier.fit(features, encoded_labels)

        train_preds = self._classifier.predict(features)
        train_f1 = f1_score(encoded_labels, train_preds, average="weighted")
        elapsed = time.time() - t0

        print(f"✅ Trained in {elapsed:.1f}s. Train F1 (weighted): {train_f1:.4f}")
        self._fitted = True
        return self

    def predict(self, texts: list[str]) -> list[str]:
        """Return string labels for a list of input texts."""
        if self.strategy == "zero_shot":
            return self._zero_shot_predict(texts)

        if self.strategy == "few_shot_llm":
            return self._few_shot_predict(texts)

        self._assert_fitted()
        features = self._extract_features(texts, fit=False)
        encoded = self._classifier.predict(features)
        return self._label_encoder.inverse_transform(encoded).tolist()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return class probabilities (not available for SVM or zero-shot)."""
        self._assert_fitted()
        if not hasattr(self._classifier, "predict_proba"):
            raise ValueError(f"predict_proba is not supported for this classifier type.")
        features = self._extract_features(texts, fit=False)
        return self._classifier.predict_proba(features)

    def evaluate(self, texts: list[str], labels: list[str]) -> dict:
        """Print a classification report and return F1 metrics."""
        preds = self.predict(texts)
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        acc = accuracy_score(labels, preds)
        print(classification_report(labels, preds, zero_division=0))
        metrics = {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4)}
        print("📊 Evaluation:", metrics)
        return metrics

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self, path)
        print(f"💾 TextClassifier saved → {path}")

    @classmethod
    def load(cls, path: str) -> "TextClassifier":
        clf = joblib.load(path)
        print(f"📂 TextClassifier loaded ← {path}")
        return clf

    # ------------------------------------------------------------------
    # Internal feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, texts: list[str], fit: bool = False) -> np.ndarray:
        if self.strategy == "embeddings":
            if self._embedding_model is None:
                print(f"⚙️  Loading embeddings model: {self.embedding_model_name}...")
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
            return self._embedding_model.encode(texts, show_progress_bar=False)

        else:  # tfidf
            if fit:
                from sklearn.feature_extraction.text import TfidfVectorizer
                self._tfidf = TfidfVectorizer(
                    max_features=10_000,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                )
                return self._tfidf.fit_transform(texts).toarray()
            else:
                return self._tfidf.transform(texts).toarray()

    def _setup_zero_shot(self):
        if not _HAS_TRANSFORMERS:
            raise RuntimeError("transformers not installed.")
        print(f"⚙️  Loading zero-shot model: {self.zero_shot_model_name}...")
        self._zero_shot_pipe = hf_pipeline(
            "zero-shot-classification",
            model=self.zero_shot_model_name,
        )
        self._fitted = True

    def _zero_shot_predict(self, texts: list[str]) -> list[str]:
        if self._zero_shot_pipe is None:
            self._setup_zero_shot()
        if not self.candidate_labels:
            raise ValueError("Provide candidate_labels for zero-shot classification.")
        results = self._zero_shot_pipe(texts, self.candidate_labels, multi_label=False)
        return [r["labels"][0] for r in results]

    def _few_shot_predict(self, texts: list[str]) -> list[str]:
        """Use the factory LLM with in-context examples."""
        from tasks.language.factory import get_llm
        llm = get_llm()
        examples = getattr(self, "_few_shot_examples", [])
        example_str = "\n".join([f"Text: {t}\nLabel: {l}" for t, l in examples])
        labels_str = ", ".join(self.candidate_labels) if self.candidate_labels else "unknown"

        preds = []
        for text in texts:
            prompt = (
                f"Classify the following text into one of these labels: {labels_str}.\n\n"
                f"Examples:\n{example_str}\n\n"
                f"Text: {text}\n"
                f"Label (respond with only the label):"
            )
            response = llm.invoke(prompt).content.strip()
            preds.append(response)
        return preds

    def _assert_fitted(self):
        if not self._fitted:
            raise RuntimeError("Classifier is not fitted. Call .fit() first.")


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    texts = [
        "The patient has severe chest pain and is struggling to breathe.",
        "Minor bruise on right knee, no swelling, normal vitals.",
        "Unconscious, no pulse. Emergency resuscitation required.",
        "Patient reports mild headache for two days.",
        "Allergic reaction with swollen face and difficulty breathing.",
    ]
    labels = ["urgent", "non-urgent", "urgent", "non-urgent", "urgent"]

    # Test TF-IDF (no heavy dependencies)
    clf = TextClassifier(strategy="tfidf")
    clf.fit(texts, labels)
    preds = clf.predict(texts)
    print(f"\nPredictions: {preds}")
    clf.evaluate(texts, labels)
    print("✅ TextClassifier smoke test passed.")
