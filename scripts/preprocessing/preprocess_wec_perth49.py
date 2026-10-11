#!/usr/bin/env python3
"""
Preprocess WEC Perth 49 for MTP experiments.

Assumption for this experiment:
- Each row is one alternative/configuration.
- Power1 ... Power49 are the 49 utility dimensions.
- Larger power is preferred in every dimension (nonnegative linear utilities).
- Coordinates (X1/Y1 ... X49/Y49), qW, and Total_Power are excluded.
- An explicit `id` column is added so the MTP C++ program preserves source row IDs.

The script does NOT normalize values; the current MTP C++ implementation
normalizes the utility dimensions internally.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Create a 49-dimensional MTP CSV from WEC Perth 49."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="WEC_Perth_49(1).csv",
        help="Source CSV (default: WEC_Perth_49(1).csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="wec_perth_49_mtp.csv",
        help="Output CSV (default: wec_perth_49_mtp.csv)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output)

    if not input_path.is_file():
        sys.exit(
            f"ERROR: input file not found: {input_path}\n"
            "Pass the source CSV path as the first argument."
        )

    print("Reading source dataset...")
    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        sys.exit(f"ERROR: could not read CSV: {exc}")

    # Normalize header whitespace but otherwise keep the source column names.
    df.columns = [str(c).strip() for c in df.columns]

    power_columns = [f"Power{i}" for i in range(1, 50)]
    missing_columns = [c for c in power_columns if c not in df.columns]
    if missing_columns:
        sys.exit(
            "ERROR: expected all Power1...Power49 columns, but these are missing: "
            + ", ".join(missing_columns)
        )

    print(f"Input rows: {len(df)}")
    print(f"Input columns: {len(df.columns)}")

    # Convert selected utility columns to numeric; invalid values become NaN
    # and are reported before rows are removed.
    powers = df[power_columns].apply(pd.to_numeric, errors="coerce")
    nonfinite_mask = ~np.isfinite(powers.to_numpy(dtype=float))
    invalid_rows = nonfinite_mask.any(axis=1)
    invalid_count = int(invalid_rows.sum())

    if invalid_count:
        print(
            f"Rows with missing/non-numeric/non-finite power values: {invalid_count}"
        )
        print("These rows will be excluded from the MTP dataset.")
    else:
        print("Power columns: all values are numeric and finite.")

    valid_powers = powers.loc[~invalid_rows].copy()
    source_positions = np.flatnonzero(~invalid_rows)

    if valid_powers.empty:
        sys.exit("ERROR: no valid rows remain after checking power columns.")

    # Verify dimensions have variation; constant dimensions are retained for
    # transparent, reproducible preprocessing, but explicitly reported.
    constant_columns = [
        c for c in power_columns if valid_powers[c].nunique(dropna=False) <= 1
    ]
    if constant_columns:
        print("WARNING: constant power dimensions detected:")
        print(", ".join(constant_columns))
        print("They are retained in the output; review whether to remove them.")

    # Compare the supplied Total_Power with the sum of Power1...Power49 as a
    # diagnostic only. It is not used as an additional utility dimension.
    if "Total_Power" in df.columns:
        total = pd.to_numeric(df.loc[valid_powers.index, "Total_Power"], errors="coerce")
        sums = valid_powers.sum(axis=1)
        comparable = np.isfinite(total.to_numpy(dtype=float))
        if comparable.any():
            max_abs_diff = float(
                np.max(np.abs(total.to_numpy(dtype=float)[comparable] -
                              sums.to_numpy(dtype=float)[comparable]))
            )
            print(f"Max |Total_Power - sum(Power1...Power49)|: {max_abs_diff:.6g}")
            if max_abs_diff > 1e-3:
                print(
                    "NOTE: Total_Power is not exactly the sum of the 49 power "
                    "columns (or differs due to rounding/definition)."
                )

    # Keep the original zero-based source row position as the stable tuple ID.
    # Invalid rows are omitted, so IDs can have gaps; this is intentional.
    output_df = pd.DataFrame({"id": source_positions})
    for column in power_columns:
        output_df[column] = valid_powers[column].to_numpy(dtype=float)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, float_format="%.10g")

    print("\nPreprocessing summary")
    print("---------------------")
    print(f"Rows written: {len(output_df)}")
    print(f"Utility dimensions: {len(power_columns)}")
    print(f"Excluded columns: coordinates X1/Y1...X49/Y49, qW, Total_Power")
    print(f"Output: {output_path.resolve()}")
    print("\nFirst 3 output rows:")
    print(output_df.head(3).to_string(index=False))
    print("\nDone. No normalization was applied; the MTP program normalizes internally.")


if __name__ == "__main__":
    main()
