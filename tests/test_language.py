"""
tests/test_language.py
------------------------
Unit tests for the language task module (RAG + TextClassifier).

Run:
    pytest tests/test_language.py -v
"""

import pytest
import numpy as np

from tasks.language.classifier import TextClassifier


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def medical_texts():
    return [
        "Patient exhibits severe chest pain and shortness of breath.",
        "Minor bruise on right knee, no swelling, normal vitals.",
        "Unconscious, no pulse. Emergency resuscitation required.",
        "Patient reports mild headache for two days, afebrile.",
        "Anaphylaxis: generalised urticaria, stridor, hypotension.",
        "Follow-up for diabetes management, blood glucose stable.",
        "Acute MI suspected, ST elevation in leads II, III, aVF.",
        "Child with mild fever 38.1°C, no rash, eating normally.",
    ]


@pytest.fixture
def medical_labels():
    return ["urgent", "non-urgent", "urgent", "non-urgent",
            "urgent", "non-urgent", "urgent", "non-urgent"]


# ===========================================================================
# TextClassifier — TF-IDF (always available, no heavy dependencies)
# ===========================================================================

def test_tfidf_fit_predict(medical_texts, medical_labels):
    clf = TextClassifier(strategy="tfidf")
    clf.fit(medical_texts, medical_labels)
    preds = clf.predict(medical_texts)
    assert len(preds) == len(medical_texts)
    assert all(p in {"urgent", "non-urgent"} for p in preds)


def test_tfidf_evaluate(medical_texts, medical_labels):
    clf = TextClassifier(strategy="tfidf")
    clf.fit(medical_texts, medical_labels)
    metrics = clf.evaluate(medical_texts, medical_labels)
    assert "accuracy" in metrics
    assert "f1_weighted" in metrics
    # Should overfit perfectly on same-data evaluation
    assert metrics["accuracy"] >= 0.5


def test_tfidf_single_text(medical_texts, medical_labels):
    """Single-item list must work (no shape errors)."""
    clf = TextClassifier(strategy="tfidf")
    clf.fit(medical_texts, medical_labels)
    preds = clf.predict(["Patient is unconscious and not breathing."])
    assert len(preds) == 1


def test_tfidf_multiclass(medical_texts):
    """Test with more than 2 classes."""
    labels = ["trauma", "cardiac", "respiratory", "neuro",
              "cardiac", "neuro", "cardiac", "respiratory"]
    clf = TextClassifier(strategy="tfidf")
    clf.fit(medical_texts, labels)
    preds = clf.predict(medical_texts)
    assert len(preds) == len(medical_texts)
    assert all(p in {"trauma", "cardiac", "respiratory", "neuro"} for p in preds)


def test_classifier_predict_before_fit_raises(medical_texts):
    clf = TextClassifier(strategy="tfidf")
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict(medical_texts)


# ===========================================================================
# TextClassifier — Persistence
# ===========================================================================

def test_save_load_roundtrip(medical_texts, medical_labels, tmp_path):
    clf = TextClassifier(strategy="tfidf")
    clf.fit(medical_texts, medical_labels)
    preds_before = clf.predict(medical_texts)

    path = str(tmp_path / "clf.pkl")
    clf.save(path)
    loaded = TextClassifier.load(path)
    preds_after = loaded.predict(medical_texts)

    assert preds_before == preds_after


# ===========================================================================
# TextClassifier — Strategy fallback
# ===========================================================================

def test_embeddings_falls_back_gracefully(medical_texts, medical_labels, monkeypatch):
    """
    If sentence-transformers is not installed, strategy should fall back to tfidf
    without crashing.
    """
    import tasks.language.classifier as clf_module
    # Temporarily simulate sentence-transformers being absent
    monkeypatch.setattr(clf_module, "_HAS_SENTENCE_TRANSFORMERS", False)

    with pytest.warns(UserWarning, match="sentence-transformers"):
        clf = TextClassifier(strategy="embeddings")

    assert clf.strategy == "tfidf"
    clf.fit(medical_texts, medical_labels)
    preds = clf.predict(medical_texts)
    assert len(preds) == len(medical_texts)


def test_zero_shot_falls_back_gracefully(medical_texts, medical_labels, monkeypatch):
    """If transformers is not installed, zero_shot should fall back to tfidf."""
    import tasks.language.classifier as clf_module
    monkeypatch.setattr(clf_module, "_HAS_TRANSFORMERS", False)

    with pytest.warns(UserWarning, match="transformers"):
        clf = TextClassifier(strategy="zero_shot")

    assert clf.strategy == "tfidf"


# ===========================================================================
# RAGPipeline — import + initialization (no real LLM required)
# ===========================================================================

def test_rag_pipeline_import():
    """Verify RAGPipeline can be imported without errors."""
    from tasks.language.rag import RAGPipeline
    assert RAGPipeline is not None


def test_rag_pipeline_instantiation(monkeypatch, tmp_path):
    """
    RAGPipeline.__init__ calls get_llm() and get_embeddings() from the factory.
    We mock these to avoid needing real API credentials during testing.
    """
    from unittest.mock import MagicMock, patch
    mock_llm = MagicMock()
    mock_embeddings = MagicMock()

    with patch("tasks.language.rag.get_llm", return_value=mock_llm), \
         patch("tasks.language.rag.get_embeddings", return_value=mock_embeddings):
        from tasks.language.rag import RAGPipeline
        rag = RAGPipeline(persist_directory=str(tmp_path / "chroma"))
        assert rag.llm is mock_llm
        assert rag.embeddings is mock_embeddings
