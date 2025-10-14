#!/usr/bin/env python3
"""
Pulsar Dataset Generator

Generates a synthetic dataset that mimics the HTRU2 Pulsar Stars dataset's
shape and rough statistical behavior for experimentation and reproducibility.
The output CSV is saved to `Lab 06/Data/pulsar_stars_synthetic.csv`.
"""

import os
import argparse
import numpy as np
import pandas as pd


def generate_synthetic_pulsar_like(num_samples: int, positive_ratio: float, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    num_positive = int(num_samples * positive_ratio)
    num_negative = num_samples - num_positive

    # 8 features; positives are slightly shifted to be more separable
    X_neg = rng.normal(loc=0.0, scale=1.0, size=(num_negative, 8))
    X_pos = rng.normal(loc=0.5, scale=1.0, size=(num_positive, 8))

    X = np.vstack([X_neg, X_pos])
    y = np.hstack([np.zeros(num_negative, dtype=int), np.ones(num_positive, dtype=int)])

    columns = [
        " Mean of the integrated profile",
        " Standard deviation of the integrated profile",
        " Excess kurtosis of the integrated profile",
        " Skewness of the integrated profile",
        " Mean of the DM-SNR curve",
        " Standard deviation of the DM-SNR curve",
        " Excess kurtosis of the DM-SNR curve",
        " Skewness of the DM-SNR curve",
    ]

    df = pd.DataFrame(X, columns=columns)
    df["target_class"] = y
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic pulsar-like dataset")
    parser.add_argument("--num-samples", type=int, default=17898, help="Total number of samples to generate")
    parser.add_argument("--positive-ratio", type=float, default=0.092, help="Ratio of positive class (pulsars)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    lab_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(lab_dir, "Data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "pulsar_stars_synthetic.csv")

    df = generate_synthetic_pulsar_like(args.num_samples, args.positive_ratio, args.seed)
    df.to_csv(out_path, index=False)

    print("Synthetic dataset generated.")
    print(f"Path: {out_path}")
    print(f"Shape: {df.shape}")
    print(df["target_class"].value_counts().sort_index())


if __name__ == "__main__":
    main()


