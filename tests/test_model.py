import json
from pathlib import Path

from app.config import get_settings
from app.model import SpamModel, explain_signals

ROOT = Path(__file__).resolve().parent.parent


def _model():
    s = get_settings()
    return SpamModel.load(s.model_path, s.metrics_path, s.review_floor)


def test_verdict_bands_are_ordered():
    m = _model()
    assert 0 < m.review_floor <= m.threshold < 1


def test_batch_matches_single():
    m = _model()
    texts = ["Win a free prize now!!! Call 08001234567", "Lunch at noon?"]
    batch = [p.score for p in m.predict(texts)]
    single = [m.predict([t])[0].score for t in texts]
    assert batch == single


def test_signals():
    sig = explain_signals("FREE £500 prize!!! Click here http://x.co NOW URGENT WINNER CLAIM TODAY")
    assert {"contains_url", "mentions_money", "urgency_or_prize_wording", "many_exclamation_marks"} <= set(sig)
    assert explain_signals("see you at lunch") == []


def test_quality_gate_on_shipped_metrics():
    """Fails if someone ships a retrained model that regressed."""
    t = json.loads((ROOT / "models" / "metrics.json").read_text())["held_out_test"]
    assert t["precision"] >= 0.97, "too many real messages would be flagged as spam"
    assert t["recall"] >= 0.85
    assert t["pr_auc"] >= 0.97
