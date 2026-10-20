import json
import os
from pathlib import Path

SPAM = "URGENT! You have won a 1000 pound prize. Call 09061701461 now or text CLAIM to 81010 to collect."
HAM = "Hi Priya, can we move tomorrow's design review to 3pm? I will share the slides before then."


def test_health_and_ready(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json()["status"] == "ready"


def test_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Spam Classifier" in r.text


def test_predict_requires_key(client):
    assert client.post("/v1/predict", json={"text": SPAM}).status_code == 401
    bad = client.post("/v1/predict", json={"text": SPAM}, headers={"X-API-Key": "nope"})
    assert bad.status_code == 401


def test_predict_spam_and_ham(client, auth):
    spam = client.post("/v1/predict", json={"text": SPAM}, headers=auth).json()
    assert spam["verdict"] == "spam" and spam["is_spam"] is True
    assert spam["score"] > spam["threshold"]
    assert "contains_phone_number" in spam["signals"]

    ham = client.post("/v1/predict", json={"subject": "Design review", "body": HAM}, headers=auth).json()
    assert ham["verdict"] == "ham" and ham["is_spam"] is False
    assert 0.0 <= ham["score"] < 0.35


def test_request_id_roundtrip(client, auth):
    r = client.post("/v1/predict", json={"text": HAM}, headers={**auth, "X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"
    assert r.json()["request_id"] == "abc123"


def test_validation(client, auth):
    assert client.post("/v1/predict", json={}, headers=auth).status_code == 422
    assert client.post("/v1/predict", json={"text": "   "}, headers=auth).status_code == 422
    too_long = client.post("/v1/predict", json={"text": "a" * 25_000}, headers=auth)
    assert too_long.status_code == 413
    over_field = client.post("/v1/predict", json={"text": "a" * 60_000}, headers=auth)
    assert over_field.status_code == 422


def test_batch(client, auth):
    r = client.post("/v1/predict/batch", json={"items": [{"text": SPAM}, {"text": HAM}]}, headers=auth)
    assert r.status_code == 200
    verdicts = [x["verdict"] for x in r.json()["results"]]
    assert verdicts == ["spam", "ham"]


def test_batch_limits(client, auth):
    assert client.post("/v1/predict/batch", json={"items": []}, headers=auth).status_code == 422
    big = {"items": [{"text": "hello"}] * 101}
    assert client.post("/v1/predict/batch", json=big, headers=auth).status_code == 413


def test_feedback_stored_once(client, auth):
    body = {"text": "unique feedback sample 7731", "correct_label": "spam"}
    first = client.post("/v1/feedback", json=body, headers=auth).json()
    second = client.post("/v1/feedback", json=body, headers=auth).json()
    assert first["stored"] is True and second["stored"] is False
    lines = Path(os.environ["SPAM_FEEDBACK_PATH"]).read_text().splitlines()
    assert json.loads(lines[-1])["label"] == 1


def test_feedback_requires_key(client):
    r = client.post("/v1/feedback", json={"text": "x", "correct_label": "ham"})
    assert r.status_code == 401


def test_demo_endpoint_open_but_rate_limited(client):
    for _ in range(5):
        assert client.post("/demo/predict", json={"text": HAM}).status_code == 200
    assert client.post("/demo/predict", json={"text": HAM}).status_code == 429


def test_model_info_reports_measured_metrics(client):
    info = client.get("/v1/model").json()
    assert info["held_out_test"]["precision"] > 0.9
    assert "SMS" in info["caveat"]


def test_metrics_endpoint_protected(client, auth):
    assert client.get("/metrics").status_code == 401
    r = client.get("/metrics", headers=auth)
    assert r.status_code == 200 and "spam_requests_total" in r.text


def test_security_headers(client):
    r = client.get("/healthz")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
