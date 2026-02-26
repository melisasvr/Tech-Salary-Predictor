"""
Data Cleaning & Feature Engineering Pipeline
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

os.makedirs("models", exist_ok=True)


def load_and_clean(path: str = "data/salaries.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # ── 1. Basic cleaning ───────────────────────────────────────────────────
    df = df.drop_duplicates()
    df = df.dropna(subset=["total_compensation", "base_salary"])

    # Cap outliers at 1st/99th percentile
    for col in ["base_salary", "total_compensation", "annual_bonus", "annual_stock"]:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    # ── 2. Feature Engineering ──────────────────────────────────────────────
    # Seniority level from role title
    def seniority(role):
        r = role.lower()
        if "principal" in r: return 5
        if "staff" in r:     return 4
        if "senior" in r or "manager" in r: return 3
        if "ml" in r or "data scientist" in r: return 2
        return 1
    df["seniority_level"] = df["role"].apply(seniority)

    # Is FAANG?
    df["is_faang"] = (df["company_tier"] == "FAANG").astype(int)

    # Is big tech (FAANG + Tier2)?
    df["is_big_tech"] = df["company_tier"].isin(["FAANG", "Tier2"]).astype(int)

    # YOE buckets
    df["yoe_bucket"] = pd.cut(
        df["years_of_experience"],
        bins=[-1, 2, 5, 10, 15, 100],
        labels=["0-2", "3-5", "6-10", "11-15", "15+"]
    )

    # Log transform target (reduces right skew)
    df["log_total_comp"] = np.log1p(df["total_compensation"])

    return df


def encode_features(df: pd.DataFrame, fit: bool = True):
    """
    Returns (X, y, encoders_dict)
    If fit=True, fits new encoders and saves them.
    If fit=False, loads saved encoders.
    """
    cat_cols = ["role", "location", "education", "company_tier", "yoe_bucket"]
    num_cols = ["years_of_experience", "seniority_level", "is_faang",
                "is_big_tech", "remote_work"]

    encoders = {}

    if fit:
        for col in cat_cols:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        joblib.dump(encoders, "models/encoders.pkl")
    else:
        encoders = joblib.load("models/encoders.pkl")
        for col in cat_cols:
            le = encoders[col]
            df[col + "_enc"] = df[col].apply(
                lambda x: le.transform([str(x)])[0]
                if str(x) in le.classes_ else -1
            )

    feature_cols = [c + "_enc" for c in cat_cols] + num_cols
    X = df[feature_cols]
    y = df["log_total_comp"]

    return X, y, encoders, feature_cols


if __name__ == "__main__":
    df = load_and_clean()
    X, y, encoders, feature_cols = encode_features(df, fit=True)
    print(f"✅ Features: {feature_cols}")
    print(f"   X shape: {X.shape}, y shape: {y.shape}")
    print(f"   Saved encoders → models/encoders.pkl")