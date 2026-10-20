# Spam Classifier

A deployable spam-detection service with a REST API, a web UI, and an n8n workflow that triages a Gmail inbox automatically.

**What makes it production-shaped:** API-key auth, per-IP rate limiting, request IDs, structured JSON logs (message text is never logged), Prometheus metrics, health/readiness probes, input limits, a human-feedback loop for retraining, tests, CI that retrains from source, and a Docker image.

## Measured performance

Evaluated once on a held-out 20% split (stratified, duplicates removed *before* splitting so no message appears in both sets). Model chosen by 5-fold cross-validation on the training split only; decision threshold tuned for precision, because hiding a real email is worse than missing one spam.

| Metric | Held-out test (1032 messages) |
|---|---|
| Precision | 99.2% |
| Recall | 91.4% |
| F1 | 0.951 |
| PR-AUC | 0.993 |
| False positives / false negatives | 1 / 11 |

Selected model: `linear_svm_calibrated` (word and character TF-IDF, calibrated linear SVM). Naive Bayes and logistic regression were compared; see `models/metrics.json`.

**Limits, stated plainly:** the dataset is the UCI SMS Spam Collection (short English texts, ~5k rows). Long marketing emails and non-English mail are out of distribution. For real email, retrain on your own labelled mail: `python -m training.train --data mail.csv --text-col body --label-col label`. Messages scoring in the "review" band are flagged for a human, not auto-actioned.

## Architecture

```
Gmail ──> n8n ──POST /v1/predict──> FastAPI ──> sklearn pipeline
            │                          │
            ├─ spam   -> move to Spam   ├─ /metrics (Prometheus)
            └─ review -> label + Slack  └─ /v1/feedback -> data/feedback.jsonl -> retrain
```

Verdicts: `spam` (score >= threshold), `review` (borderline band), `ham`.

## Run locally

```bash
pip install -r requirements-dev.txt
python -m training.train      # writes models/spam_model.joblib and models/metrics.json
pytest -q
SPAM_API_KEYS=dev-key uvicorn app.main:app --reload
```
Open http://localhost:8000 for the UI, http://localhost:8000/docs for the API docs.

```bash
curl -X POST localhost:8000/v1/predict -H 'X-API-Key: dev-key' -H 'Content-Type: application/json' \
  -d '{"subject":"Prize","body":"WIN a FREE prize now, call 09061701461"}'
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `SPAM_API_KEYS` | empty | Comma-separated keys for `/v1/*` and `/metrics`. **Set in production.** Empty disables auth and logs a warning. |
| `SPAM_DEMO_ENABLED` | `true` | Serve the unauthenticated UI endpoint (`/demo/predict`, 20 req/min/IP). |
| `SPAM_RATE_LIMIT_API` | `300/minute` | Per-IP limit on `/v1/*`. |
| `SPAM_REVIEW_FLOOR` | `0.35` | Lower bound of the "review" band. |
| `SPAM_MAX_BATCH_SIZE` | `100` | Max items per batch call. |
| `SPAM_CORS_ORIGINS` | empty | Comma-separated allowed origins. |
| `SPAM_LOG_LEVEL` | `INFO` | Log level. |

## Deploy

**Render:** New + > Blueprint, pick this repo. `render.yaml` builds the Dockerfile and generates `SPAM_API_KEYS`. Free instances sleep when idle; use a paid plan or an uptime pinger for always-on.

**Anywhere with Docker:** `docker build -t spam-classifier . && docker run -p 8000:8000 -e SPAM_API_KEYS=change-me spam-classifier`

Feedback is written to `data/feedback.jsonl` inside the container. Mount a volume at `/srv/data` if you want it to survive restarts.

## Gmail automation (n8n)

Import `n8n/gmail_spam_triage.json` into n8n and set these variables: `SPAM_API_URL`, `SPAM_API_KEY`, `REVIEW_LABEL_ID`, `SLACK_CHANNEL`. Connect Gmail and Slack credentials in n8n. The workflow polls unread inbox mail every minute, classifies it, moves spam, and labels borderline mail and pings Slack. Test on a spare mailbox first; the workflow file has not been run against a live Gmail account.

## Retraining with feedback

`POST /v1/feedback` stores human corrections (deduplicated by hash). Append them to your training CSV and rerun `python -m training.train`. CI retrains from source on every push, so the committed model cannot silently drift from the code.

## Project layout

```
app/        FastAPI service (main, config, model, schemas, static UI)
training/   train.py: cross-validated training and evaluation
models/     spam_model.joblib, metrics.json
tests/      API, model, and training tests (20)
n8n/        Gmail triage workflow
```

MIT licensed.
