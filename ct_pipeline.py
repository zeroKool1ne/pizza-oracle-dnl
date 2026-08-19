"""Pizza Oracle Continuous Training pipeline.

Flow:
 1. Download latest data + champion model from S3
 2. Evaluate champion on the most RECENT slice of data
 3. If degradation > threshold (or no champion exists) -> retrain a challenger
 4. Challenger must BEAT the champion on recent data to be promoted
 5. Every run is logged to metrics/history.jsonl (audit trail)

Design choice: we retrain on a SLIDING WINDOW (the last TRAIN_WINDOW_ROWS rows)
instead of all history. After concept drift, old-regime data actively poisons
the model — forgetting is a feature. Discuss the trade-offs with your instructor.
"""
import io
import json
import os
import sys
from datetime import datetime, timezone

import boto3
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

BUCKET = os.environ["BUCKET"]
DEGRADATION_FACTOR = 1.5   # retrain if recent RMSE > baseline RMSE * 1.5
RECENT_FRACTION = 0.2      # last 20% of rows = "recent window" (evaluation)
TRAIN_WINDOW_ROWS = 500    # sliding window: train only on the freshest rows
FEATURES = ["temperature", "is_rainy", "is_weekend", "promo"]
TARGET = "orders"

s3 = boto3.client("s3", region_name="us-east-1")


def load_data() -> pd.DataFrame:
    obj = s3.get_object(Bucket=BUCKET, Key="data/orders.csv")
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def load_champion():
    """Returns (model, metrics) or (None, None) if no champion exists yet."""
    try:
        m = s3.get_object(Bucket=BUCKET, Key="models/champion.pkl")
        model = joblib.load(io.BytesIO(m["Body"].read()))
        j = s3.get_object(Bucket=BUCKET, Key="metrics/champion.json")
        metrics = json.loads(j["Body"].read())
        return model, metrics
    except s3.exceptions.NoSuchKey:
        return None, None


def train(df: pd.DataFrame):
    window = df.tail(TRAIN_WINDOW_ROWS)
    model = LinearRegression()
    model.fit(window[FEATURES], window[TARGET])
    return model


def rmse_on(model, df: pd.DataFrame) -> float:
    preds = model.predict(df[FEATURES])
    return float(root_mean_squared_error(df[TARGET], preds))


def promote(model, recent_rmse: float, rows: int):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    buf = io.BytesIO()
    joblib.dump(model, buf)
    body = buf.getvalue()
    s3.put_object(Bucket=BUCKET, Key="models/champion.pkl", Body=body)
    s3.put_object(Bucket=BUCKET, Key=f"models/archive/model-{ts}.pkl", Body=body)
    metrics = {
        "baseline_rmse": recent_rmse,
        "trained_at": ts,
        "trained_on_rows": rows,
        "coefficients": dict(zip(FEATURES, [round(c, 3) for c in model.coef_])),
    }
    s3.put_object(
        Bucket=BUCKET, Key="metrics/champion.json",
        Body=json.dumps(metrics, indent=2),
    )
    return metrics


def log_run(entry: dict):
    try:
        old = s3.get_object(Bucket=BUCKET, Key="metrics/history.jsonl")["Body"].read().decode()
    except s3.exceptions.NoSuchKey:
        old = ""
    s3.put_object(
        Bucket=BUCKET, Key="metrics/history.jsonl",
        Body=old + json.dumps(entry) + "\n",
    )


def main():
    run_ts = datetime.now(timezone.utc).isoformat()
    df = load_data()
    recent = df.tail(int(len(df) * RECENT_FRACTION))
    print(f"📦 Dataset: {len(df)} rows | recent window: {len(recent)} rows")

    champion, champ_metrics = load_champion()
    entry = {"ts": run_ts, "rows": len(df)}

    if champion is None:
        print("👶 No champion found — training the first model (cold start).")
        model = train(df)
        metrics = promote(model, rmse_on(model, recent), min(len(df), TRAIN_WINDOW_ROWS))
        entry.update({"action": "cold_start", "new_baseline_rmse": metrics["baseline_rmse"]})
        log_run(entry)
        print(f"🏆 First champion promoted! Coefficients: {metrics['coefficients']}")
        return

    baseline = champ_metrics["baseline_rmse"]
    current = rmse_on(champion, recent)
    threshold = baseline * DEGRADATION_FACTOR
    print(f"🔍 Champion RMSE on recent data: {current:.2f} "
          f"(baseline: {baseline:.2f}, retrain threshold: {threshold:.2f})")
    entry.update({"champion_rmse_recent": round(current, 2),
                  "baseline_rmse": round(baseline, 2)})

    if current <= threshold:
        print("😴 No significant drift. Champion keeps the crown. Going back to sleep.")
        entry["action"] = "no_drift"
        log_run(entry)
        return

    print("🚨 DRIFT DETECTED! Training a challenger on fresh data...")
    challenger = train(df)
    challenger_rmse = rmse_on(challenger, recent)
    print(f"🥊 Challenger RMSE on recent data: {challenger_rmse:.2f} "
          f"vs champion: {current:.2f}")

    if challenger_rmse < current:
        metrics = promote(challenger, challenger_rmse, min(len(df), TRAIN_WINDOW_ROWS))
        entry.update({"action": "retrained_and_promoted",
                      "challenger_rmse": round(challenger_rmse, 2)})
        print(f"🏆 Challenger PROMOTED! New coefficients: {metrics['coefficients']}")
    else:
        entry.update({"action": "retrained_not_promoted",
                      "challenger_rmse": round(challenger_rmse, 2)})
        print("🛡️ Challenger did not beat the champion. Keeping the old model. "
              "(This guardrail prevents bad retrains from reaching production!)")

    log_run(entry)


if __name__ == "__main__":
    sys.exit(main())