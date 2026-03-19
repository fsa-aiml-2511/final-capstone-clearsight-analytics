#!/usr/bin/env python3
"""
Bulk Test Script — Run model predictions and validate output
=============================================================
Runs your model's predict.py on the instructor test data and validates
that the output CSV matches the required format.

DO NOT MODIFY THIS SCRIPT.

Usage:
    python bulk_test.py --model 1      # Test a single model (1-5)
    python bulk_test.py --all           # Test all 5 models sequentially
"""
import argparse
import subprocess
import sys

import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Model Configuration
# ─────────────────────────────────────────────────────────────

MODEL_CONFIG = {
    1: {
        "name": "Traditional ML",
        "dir": Path("models/model1_traditional_ml"),
        "output": Path("test_data/model1_results.csv"),
        "columns": ["id", "prediction", "probability", "confidence"],
        "timeout": 300,
    },
    2: {
        "name": "Deep Learning",
        "dir": Path("models/model2_deep_learning"),
        "output": Path("test_data/model2_results.csv"),
        "columns": ["id", "prediction", "probability", "confidence"],
        "timeout": 300,
    },
    3: {
        "name": "CNN",
        "dir": Path("models/model3_cnn"),
        "output": Path("test_data/model3_results.csv"),
        "columns": ["image_id", "predicted_class", "confidence"],
        "timeout": 600,
    },
    4: {
        "name": "NLP Classification",
        "dir": Path("models/model4_nlp_classification"),
        "output": Path("test_data/model4_results.csv"),
        "columns": ["id", "predicted_class", "confidence"],
        "timeout": 300,
    },
    5: {
        "name": "Innovation",
        "dir": Path("models/model5_innovation"),
        "output": Path("test_data/model5_results.csv"),
        "columns": ["id", "prediction", "confidence", "metric_name", "metric_value"],
        "timeout": 300,
    },
}

# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────


def validate_output(output_path: Path, required_columns: list[str]) -> bool:
    """Validate that output matches the required template format."""
    if not output_path.exists():
        print(f"  FAIL: Output file not found at {output_path}")
        return False

    try:
        df = pd.read_csv(output_path)
    except Exception as e:
        print(f"  FAIL: Could not read output CSV: {e}")
        return False

    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        print(f"  FAIL: Missing required columns: {missing_cols}")
        print(f"    Your columns: {df.columns.tolist()}")
        print(f"    Required columns: {required_columns}")
        return False

    if len(df) == 0:
        print("  FAIL: Output file is empty (0 rows)")
        return False

    # Check for null values in key prediction columns
    for col in required_columns:
        if col in df.columns and df[col].isna().any():
            print(f"  WARNING: {df[col].isna().sum()} null values in '{col}'")

    print(f"  OK: {len(df)} predictions, columns match template")
    return True


# ─────────────────────────────────────────────────────────────
# Run a single model
# ─────────────────────────────────────────────────────────────


def run_model(model_num: int) -> bool:
    """Run a single model's predict.py and validate output. Returns True on success."""
    cfg = MODEL_CONFIG[model_num]

    print("=" * 60)
    print(f"BULK TEST — Model {model_num}: {cfg['name']}")
    print("=" * 60)

    predict_script = cfg["dir"] / "predict.py"
    if not predict_script.exists():
        print(f"  FAIL: {predict_script} not found")
        return False

    print(f"  Running: python {predict_script}")
    try:
        result = subprocess.run(
            [sys.executable, str(predict_script)],
            capture_output=True,
            text=True,
            timeout=cfg["timeout"],
        )
    except subprocess.TimeoutExpired:
        print(f"  FAIL: predict.py timed out after {cfg['timeout']}s")
        return False

    if result.returncode != 0:
        print(f"  FAIL: predict.py exited with code {result.returncode}")
        print(f"  STDERR:\n{result.stderr}")
        return False

    if result.stdout.strip():
        print(result.stdout)

    if validate_output(cfg["output"], cfg["columns"]):
        print(f"\n  SUCCESS: Results written to {cfg['output']}")
        return True
    else:
        print(f"\n  FAILED: Output validation failed")
        return False


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Bulk test script for capstone models",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Model number to test (1-5)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Test all 5 models sequentially",
    )
    args = parser.parse_args()

    if args.model:
        success = run_model(args.model)
        sys.exit(0 if success else 1)

    # --all: run all models and print summary
    results = {}
    for model_num in range(1, 6):
        results[model_num] = run_model(model_num)
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model_num in range(1, 6):
        cfg = MODEL_CONFIG[model_num]
        status = "PASS" if results[model_num] else "FAIL"
        print(f"  Model {model_num} ({cfg['name']}): {status}")

    passed = sum(results.values())
    print(f"\n  {passed}/5 models passed")
    print("=" * 60)

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
