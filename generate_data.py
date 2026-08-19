# This is the NORMAL era

"""Generate synthetic pizza order data for the Pizza Oracle.

Eras:
  normal    -> rainy days boost delivery orders, HOT days REDUCE orders
  pineapple -> the viral incident: HOT days now massively INCREASE orders
"""
import argparse
import numpy as np
import pandas as pd

def generate(era: str, rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    temperature = rng.uniform(5, 35, rows).round(1)      # °C
    is_rainy = rng.integers(0, 2, rows)
    is_weekend = rng.integers(0, 2, rows)
    promo = (rng.random(rows) < 0.3).astype(int)
    noise = rng.normal(0, 8, rows)

    base = 80 + 25 * is_weekend + 30 * promo + 20 * is_rainy

    if era == "normal":
        # Hot days: people lose appetite for pizza (-0.9 orders per °C)
        orders = base - 0.9 * temperature + noise
    elif era == "pineapple":
        # 🍍 THE INCIDENT: hot days = pineapple pizza hype (+2.5 orders per °C)
        orders = base + 2.5 * temperature + noise
    else:
        raise ValueError(f"Unknown era: {era}")

    return pd.DataFrame({
        "temperature": temperature,
        "is_rainy": is_rainy,
        "is_weekend": is_weekend,
        "promo": promo,
        "orders": orders.round(0).clip(min=0),
    })

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--era", choices=["normal", "pineapple"], required=True)
    p.add_argument("--rows", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="orders.csv")
    args = p.parse_args()

    df = generate(args.era, args.rows, args.seed)
    df.to_csv(args.out, index=False)
    print(f"✅ Generated {len(df)} rows of '{args.era}' era data -> {args.out}")
    print(df.describe().round(2))
