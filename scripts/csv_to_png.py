#!/usr/bin/env python3
"""Convert one or more CSV files to PNG table images.

Usage:
  python3 scripts/csv_to_png.py segment_report.csv anomaly_report.csv
  python3 scripts/csv_to_png.py *.csv --out-dir=pngs --dpi=200

The script formats numeric cells to 3 decimal places.
"""

import argparse
import os
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def df_to_png(df: pd.DataFrame, out_path: str, dpi: int = 200, fontsize: int = 8):
    # Prepare a display copy where numeric values are formatted to 3 decimals
    disp = df.copy()
    num_cols = disp.select_dtypes(include=["number"]).columns
    for c in num_cols:
        disp[c] = disp[c].apply(lambda x: ("{:.3f}".format(x)) if pd.notnull(x) else "")

    nrows, ncols = disp.shape
    # approximate figure size: width per column, height per row
    col_width = max(1.0, min(2.5, 0.6 * max(1, ncols)))
    row_height = 0.25
    figsize = (col_width * ncols, max(2, row_height * (nrows + 1)))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    table = ax.table(
        cellText=disp.fillna("").values,
        colLabels=list(disp.columns),
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Convert CSV files to PNG table images.")
    parser.add_argument("files", nargs='+', help="CSV files to convert")
    parser.add_argument("--out-dir", default=".", help="Output directory for PNGs")
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI")
    parser.add_argument("--fontsize", type=int, default=8, help="Table font size")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for path in args.files:
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue

        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.out_dir, f"{base}.png")
        print(f"Rendering {path} → {out_path}")
        try:
            df_to_png(df, out_path, dpi=args.dpi, fontsize=args.fontsize)
        except Exception as e:
            print(f"Failed to render {path}: {e}")


if __name__ == '__main__':
    main()
