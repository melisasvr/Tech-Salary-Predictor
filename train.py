"""
Model Training: trains GradientBoosting, RandomForest, XGBoost (if available),
picks best by CV R², saves model + metadata.
"""
import numpy as np
import pandas as pd
import joblib
import json
import os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from preprocessing import load_and_clean, encode_features

os.makedirs("models", exist_ok=True)


def train():
    # ── Load & prep ─────────────────────────────────────────────────────────
    print("📦 Loading data...")
    df = load_and_clean()
    X, y, encoders, feature_cols = encode_features(df, fit=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Candidate models ────────────────────────────────────────────────────
    candidates = {
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=5, subsample=0.8, random_state=42
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=10,
            min_samples_leaf=4, random_state=42, n_jobs=-1
        ),
    }

    try:
        from xgboost import XGBRegressor
        candidates["XGBoost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.05,
            max_depth=5, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            verbosity=0
        )
        print("✅ XGBoost detected — added to candidates")
    except ImportError:
        print("ℹ️  XGBoost not installed, skipping")

    # ── Cross-validate ──────────────────────────────────────────────────────
    best_name, best_model, best_cv = None, None, -np.inf
    results = {}

    for name, model in candidates.items():
        print(f"🔄 CV: {name}...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                     scoring="r2", n_jobs=-1)
        mean_r2 = cv_scores.mean()
        results[name] = {"cv_r2_mean": round(mean_r2, 4),
                         "cv_r2_std": round(cv_scores.std(), 4)}
        print(f"   R² = {mean_r2:.4f} ± {cv_scores.std():.4f}")
        if mean_r2 > best_cv:
            best_cv, best_name, best_model = mean_r2, name, model

    # ── Fit best on full train ───────────────────────────────────────────────
    print(f"\n🏆 Best model: {best_name}")
    best_model.fit(X_train, y_train)

    # ── Evaluate on test ─────────────────────────────────────────────────────
    y_pred_log = best_model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"   Test R²   : {r2:.4f}")
    print(f"   Test MAE  : ${mae:,.0f}")
    print(f"   Test MAPE : {mape:.1f}%")

    # ── Feature importance ───────────────────────────────────────────────────
    importances = dict(zip(feature_cols,
                           best_model.feature_importances_.tolist()))

    # ── Save artifacts ───────────────────────────────────────────────────────
    joblib.dump(best_model, "models/salary_model.pkl")

    metadata = {
        "model_name": best_name,
        "feature_cols": feature_cols,
        "cv_results": results,
        "test_metrics": {
            "r2": round(r2, 4),
            "mae": round(mae, 2),
            "mape": round(mape, 2),
        },
        "feature_importances": importances,
    }
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n✅ Saved:")
    print("   models/salary_model.pkl")
    print("   models/metadata.json")
    return best_model, metadata


if __name__ == "__main__":
    train()