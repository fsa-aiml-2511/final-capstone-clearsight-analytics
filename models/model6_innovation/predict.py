#!/usr/bin/env python3
"""
Model 6: Drug Effectiveness Recommender
========================================
Given a condition and a drug that was rated Ineffective, recommends alternative
drugs that other patients found more effective for the same condition.

Builds a ranked lookup table from the Model 4 NLP prediction output
(test_data/model4_results.csv) combined with the original patient review data.

Usage:
    python models/model6_innovation/predict.py

Output:
    test_data/model6_results.csv  - drug effectiveness rankings per condition
"""

import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.data_pipeline_wes import load_raw_data, clean_data

# =============================================================================
# Config
# =============================================================================
MODEL4_PREDICTIONS = PROJECT_ROOT / "test_data" / "model4_results.csv"
RAW_DATA_FILE      = "patient_medication_feedback.csv"
OUTPUT_FILE        = PROJECT_ROOT / "test_data" / "model6_results.csv"
MIN_REVIEWS        = 20   # minimum reviews required to include a drug in rankings


# =============================================================================
# Build effectiveness rankings
# =============================================================================

def build_rankings(df_raw, df_preds):
    """
    Join predictions with raw data, then compute per drug+condition scores.
    Returns a DataFrame ranked by effectiveness score within each condition.
    """
    df = df_raw.copy().reset_index(drop=True)
    df["predicted_class"] = df_preds["predicted_class"].values
    df["confidence"]      = df_preds["confidence"].values

    # Score: weighted % Highly Effective
    # Each review contributes its confidence score if Highly Effective, else 0
    df["is_highly_effective"]    = (df["predicted_class"] == "Highly Effective").astype(float)
    df["is_somewhat_effective"]  = (df["predicted_class"] == "Somewhat Effective").astype(float)
    df["is_ineffective"]         = (df["predicted_class"] == "Ineffective").astype(float)

    grouped = df.groupby(["condition", "urlDrugName"]).agg(
        total_reviews        = ("predicted_class", "count"),
        pct_highly_effective = ("is_highly_effective", "mean"),
        pct_somewhat_effective = ("is_somewhat_effective", "mean"),
        pct_ineffective      = ("is_ineffective", "mean"),
        avg_confidence       = ("confidence", "mean"),
    ).reset_index()

    # Filter out drug/condition combos with too few reviews to be reliable
    grouped = grouped[grouped["total_reviews"] >= MIN_REVIEWS].copy()

    # Effectiveness score: weighted sum (Highly=1.0, Somewhat=0.5, Ineffective=0.0)
    grouped["effectiveness_score"] = (
        grouped["pct_highly_effective"] * 1.0 +
        grouped["pct_somewhat_effective"] * 0.5
    )

    # Rank within each condition (1 = best)
    grouped["rank"] = grouped.groupby("condition")["effectiveness_score"].rank(
        ascending=False, method="min"
    ).astype(int)

    grouped = grouped.sort_values(["condition", "rank"])

    # Round for readability
    for col in ["pct_highly_effective", "pct_somewhat_effective", "pct_ineffective",
                "avg_confidence", "effectiveness_score"]:
        grouped[col] = grouped[col].round(4)

    return grouped


def recommend(rankings, condition, current_drug=None, top_n=5):
    """
    Print top alternative drugs for a given condition.
    Optionally excludes the current drug from results.
    """
    condition_lower = condition.lower()
    matches = rankings[rankings["condition"].str.lower() == condition_lower]

    if matches.empty:
        print(f"No data found for condition: '{condition}'")
        print("Try one of these conditions:")
        all_conditions = rankings["condition"].unique()
        similar = [c for c in all_conditions if condition_lower[:5] in c.lower()][:10]
        for c in similar:
            print(f"  - {c}")
        return

    if current_drug:
        matches = matches[matches["urlDrugName"].str.lower() != current_drug.lower()]

    top = matches.head(top_n)

    print(f"\nCondition: {condition}")
    if current_drug:
        print(f"Current drug (Ineffective): {current_drug}")
    print(f"\nTop {top_n} alternative drugs by patient-reported effectiveness:\n")
    print(f"  {'Rank':<6} {'Drug':<35} {'Highly Eff%':<14} {'Somewhat Eff%':<16} {'Ineffective%':<14} {'Reviews'}")
    print(f"  {'-'*4:<6} {'-'*33:<35} {'-'*11:<14} {'-'*13:<16} {'-'*12:<14} {'-'*7}")

    for _, row in top.iterrows():
        print(
            f"  {int(row['rank']):<6} "
            f"{row['urlDrugName']:<35} "
            f"{row['pct_highly_effective']*100:>8.1f}%     "
            f"{row['pct_somewhat_effective']*100:>8.1f}%         "
            f"{row['pct_ineffective']*100:>8.1f}%      "
            f"{int(row['total_reviews'])}"
        )
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    print("Loading data...")
    if not MODEL4_PREDICTIONS.exists():
        print(f"ERROR: Model 4 predictions not found at {MODEL4_PREDICTIONS}")
        print("Run models/model4_nlp_classification/predict.py first.")
        return

    df_preds = pd.read_csv(MODEL4_PREDICTIONS)
    df_raw   = load_raw_data(RAW_DATA_FILE)
    df_raw   = clean_data(df_raw)

    if len(df_raw) != len(df_preds):
        print(f"ERROR: Row count mismatch — raw: {len(df_raw)}, predictions: {len(df_preds)}")
        return

    print("Building drug effectiveness rankings...")
    rankings = build_rankings(df_raw, df_preds)

    # Save full rankings table
    rankings.to_csv(OUTPUT_FILE, index=False)
    print(f"Full rankings saved to {OUTPUT_FILE}")
    print(f"Total drug/condition combinations ranked: {len(rankings):,}")
    print(f"Conditions covered: {rankings['condition'].nunique():,}")
    print(f"Drugs covered: {rankings['urlDrugName'].nunique():,}")

    # Example recommendations
    print("\n" + "="*80)
    print("EXAMPLE RECOMMENDATIONS")
    print("="*80)

    recommend(rankings, "birth control", current_drug="levonorgestrel")
    recommend(rankings, "depression", current_drug="sertraline")
    recommend(rankings, "anxiety")


if __name__ == "__main__":
    main()
