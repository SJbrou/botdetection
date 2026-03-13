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
    # Compute column widths based on max content length per column
    col_max_chars = []
    for col in disp.columns:
        max_len = max([len(str(x)) for x in disp[col].fillna("").values] + [len(str(col))])
        col_max_chars.append(max_len)

    # estimate inches per character; tune to fit typical fonts
    char_width_in = 0.085
    min_col_in = 0.4
    max_col_in = 2.5
    col_widths = [min(max(char_width_in * m, min_col_in), max_col_in) for m in col_max_chars]

    row_height = 0.25
    total_width = max(6, sum(col_widths))
    total_height = max(2, row_height * (nrows + 1))
    figsize = (total_width, total_height)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    table = ax.table(
        cellText=disp.fillna("").values,
        colLabels=list(disp.columns),
        cellLoc='center',
        loc='center'
    )
    # Try to set reasonable column widths
    try:
        # matplotlib.Table supports auto setting; use our computed widths to size the figure instead
        table.auto_set_font_size(False)
    except Exception:
        pass
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
