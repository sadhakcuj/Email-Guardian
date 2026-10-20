import pandas as pd

from training.train import load_dataset, pick_threshold
import numpy as np


def test_load_dataset_dedupes_and_maps_labels(tmp_path):
    p = tmp_path / "d.csv"
    pd.DataFrame(
        {"label": ["ham", "spam", "spam", "ham"], "text": ["hi", "win now", "win now", "  "]}
    ).to_csv(p, index=False)
    df = load_dataset(p, None, None)
    assert len(df) == 2
    assert set(df["label"]) == {0, 1}


def test_pick_threshold_meets_target_precision():
    rng = np.random.default_rng(0)
    y = np.array([0] * 500 + [1] * 100)
    scores = np.concatenate([rng.beta(2, 8, 500), rng.beta(8, 2, 100)])
    thr = pick_threshold(y, scores, 0.95)
    pred = scores >= thr
    assert (y[pred] == 1).mean() >= 0.95
