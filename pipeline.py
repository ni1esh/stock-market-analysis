"""
NIFTY 50 Next-Day Open Price Prediction — Master Pipeline
Executes feature generation and automated RFE pipeline.
"""
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from features import generate_nifty_features
from train_rfe import run_rfe_pipeline

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: EXPANDING TECHNICAL INDICATORS & GENERATING DATASET")
    print("=" * 60)
    generate_nifty_features()

    print("\n" + "=" * 60)
    print("STEP 2: RUNNING RFE FEATURE SELECTION & VALIDATION OPTIMIZATION")
    print("=" * 60)
    run_rfe_pipeline()
