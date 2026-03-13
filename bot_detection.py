#!/usr/bin/env python3
"""
Standalone bot detection pipeline.
- Connects to DuckDB `data/data.duckdb`.
- Computes Python-dependent features per author from `comments`.
- Saves `user_features` table to DuckDB.
- Runs KMeans segmentation (silhouette selection) and IsolationForest anomaly detection.
- Writes outputs: `segment_report.csv`, `anomaly_report.csv`, `user_segments_and_anomalies.csv`.

Usage: python bot_detection.py [--sample N] [--chunk-size N] [--max-clusters N]
"""

import argparse
import duckdb
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from textblob import TextBlob
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def df_to_png(df: pd.DataFrame, filename: str, dpi: int = 200, fontsize: int = 8):
    """Render a DataFrame to a PNG table with tight layout and minimal whitespace.

    Numeric columns are formatted to 3 decimals.
    """
    disp = df.copy()
    # format numeric columns
    num_cols = disp.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        disp[c] = disp[c].apply(lambda x: ("{:.3f}".format(x)) if pd.notnull(x) else "")

    # compute column widths based on content length
    col_max_chars = []
    for col in disp.columns:
        max_len = max([len(str(x)) for x in disp[col].fillna("").values] + [len(str(col))])
        col_max_chars.append(max_len)

    char_width_in = 0.085
    min_col_in = 0.4
    max_col_in = 2.5
    col_widths = [min(max(char_width_in * m, min_col_in), max_col_in) for m in col_max_chars]

    nrows = max(1, len(disp))
    row_height = 0.25
    total_width = max(6, sum(col_widths))
    total_height = max(1.6, row_height * (nrows + 1))

    fig, ax = plt.subplots(figsize=(total_width, total_height))
    ax.axis('off')
    table = ax.table(
        cellText=disp.fillna("").values,
        colLabels=list(disp.columns),
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.1)
    plt.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)

# Defaults
DUCKDB_PATH = "data/data.duckdb"
TABLE_FINAL = "user_features"
CHUNK_SIZE = 100
MAX_CLUSTERS = 10
RANDOM_STATE = 42

# Feature functions

def lexical_diversity(contents):
    all_words = " ".join(contents).split()
    return len(set(all_words)) / len(all_words) if all_words else 0


def repeated_ratio(contents):
    return pd.Series(contents).duplicated().mean() if contents else 0


def link_ratio(contents):
    return np.mean([1 if "http" in (c or "").lower() else 0 for c in contents]) if contents else 0


def sentiment_polarity(contents):
    return np.mean([TextBlob(c).sentiment.polarity for c in contents]) if contents else 0


def temporal_features(group):
    timestamps = pd.to_datetime(group["creation_datetime"]).sort_values()
    n_comments = len(timestamps)
    if n_comments == 0:
        return {
            "comments_per_day": 0,
            "burstiness_hours": 0,
            "hour_entropy": 0,
        }
    account_age_days = max((timestamps.max() - timestamps.min()).days, 1)
    comments_per_day = n_comments / account_age_days
    inter_times = timestamps.diff().dt.total_seconds().dropna() / 3600
    burstiness_hours = inter_times.std() if not inter_times.empty else 0
    hours = timestamps.dt.hour
    counts = np.bincount(hours, minlength=24)
    hour_entropy_val = entropy(counts, base=2) if counts.sum() > 0 else 0
    return {
        "comments_per_day": comments_per_day,
        "burstiness_hours": burstiness_hours,
        "hour_entropy": hour_entropy_val,
    }


def compute_python_features(user_chunk_df):
    results = []
    for author_id, group in user_chunk_df.groupby("author_id"):
        contents = group["content"].fillna("").astype(str).tolist()
        temp_feats = temporal_features(group)
        results.append({
            "author_id": author_id,
            "lexical_diversity": lexical_diversity(contents),
            "repeated_ratio": repeated_ratio(contents),
            "link_ratio": link_ratio(contents),
            "sentiment_polarity": sentiment_polarity(contents),
            **temp_feats,
        })
    return pd.DataFrame(results)


def main(args):
    print("[INFO] Connecting to DuckDB...")
    con = duckdb.connect(DUCKDB_PATH)

    # Sample authors if requested
    if args.sample:
        print(f"[INFO] Sampling {args.sample} authors...")
        sampled_authors = con.execute(f"SELECT id FROM authors ORDER BY RANDOM() LIMIT {int(args.sample)}").df()["id"].tolist()
        author_filter = f"WHERE author_id IN ({','.join(map(str, sampled_authors))})"
    else:
        author_filter = ""
    
    # Year filter (optional)
    if args.year:
        year_clause = f"year = {int(args.year)}"
    else:
        year_clause = ""

    # Combine filters into a single WHERE clause for comments query
    if author_filter and year_clause:
        comment_filter = author_filter + " AND " + year_clause
    elif author_filter:
        comment_filter = author_filter
    elif year_clause:
        comment_filter = f"WHERE {year_clause}"
    else:
        comment_filter = ""

    print("[INFO] Loading comment content from DuckDB...")
    df_comments = con.execute(f"SELECT author_id, content, creation_datetime FROM comments {comment_filter}").df()
    user_count = df_comments['author_id'].nunique()
    print(f"[INFO] Found {user_count} users and {len(df_comments)} comments.")

    # Chunk authors for parallel processing
    author_ids = df_comments['author_id'].unique()
    chunk_size = args.chunk_size or CHUNK_SIZE
    author_chunks = [author_ids[i:i + chunk_size] for i in range(0, len(author_ids), chunk_size)]

    print("[INFO] Computing features in parallel...")
    all_feats = []
    with ProcessPoolExecutor(max_workers=args.workers or None) as executor:
        futures = []
        for chunk in author_chunks:
            chunk_df = df_comments[df_comments['author_id'].isin(chunk)]
            futures.append(executor.submit(compute_python_features, chunk_df))
        for future in tqdm(as_completed(futures), total=len(futures)):
            all_feats.append(future.result())

    df_feats = pd.concat(all_feats, ignore_index=True)
    print(f"[INFO] Computed features for {len(df_feats)} users.")

    # Save features to DuckDB
    print(f"[INFO] Saving features to DuckDB table {TABLE_FINAL}...")
    con.register("df_feats", df_feats)
    con.execute(f"CREATE OR REPLACE TABLE {TABLE_FINAL} AS SELECT * FROM df_feats")

    # Prepare feature matrix
    feature_columns = [
        "lexical_diversity", "repeated_ratio", "link_ratio",
        "sentiment_polarity", "comments_per_day",
        "burstiness_hours", "hour_entropy",
    ]
    X = df_feats[feature_columns].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine optimal clusters
    print("[INFO] Determining optimal number of clusters...")
    best_score = -1
    best_k = 2
    best_labels = None
    best_kmeans = None
    max_k = min(args.max_clusters or MAX_CLUSTERS, len(df_feats))
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        try:
            score = silhouette_score(X_scaled, labels)
        except Exception:
            score = -1
        print(f"  k={k}, silhouette_score={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_kmeans = kmeans

    print(f"[INFO] Selected optimal k={best_k} with silhouette score={best_score:.4f}")
    df_feats["segment"] = best_labels

    # Isolation Forest
    print("[INFO] Running Isolation Forest for anomaly detection...")
    iso_forest = IsolationForest(random_state=RANDOM_STATE, contamination=args.contamination)
    df_feats["anomaly_score"] = iso_forest.fit_predict(X_scaled)
    df_feats["anomaly_score_val"] = iso_forest.decision_function(X_scaled)

    # Exports
    print("[INFO] Generating reports and saving CSVs...")
    segment_report = df_feats.groupby("segment")[feature_columns].mean().reset_index()
    segment_counts = df_feats.groupby("segment").size().reset_index(name="count")
    segment_report = segment_report.merge(segment_counts, on="segment")
    # Round numeric values in segment report to 3 decimals, keep count as integer
    segment_report = segment_report.round(3)
    if "count" in segment_report.columns:
        try:
            segment_report["count"] = segment_report["count"].astype(int)
        except Exception:
            pass
    segment_report.to_csv("segment_report.csv", index=False)
    # Also render PNG (tight whitespace)
    try:
        df_to_png(segment_report, "segment_report.png")
    except Exception as e:
        print(f"[WARN] Failed to render segment_report.png: {e}")

    anomalies = df_feats[df_feats["anomaly_score"] == -1]
    anomaly_report = anomalies[feature_columns + ["anomaly_score_val"]].describe().transpose()
    # Round anomaly report stats to 3 decimals
    anomaly_report = anomaly_report.round(3)
    anomaly_report.to_csv("anomaly_report.csv")
    try:
        # reset index so the metric names are visible as a column
        df_to_png(anomaly_report.reset_index(), "anomaly_report.png")
    except Exception as e:
        print(f"[WARN] Failed to render anomaly_report.png: {e}")

    # Round numeric columns in the per-user output to 3 decimals for table outputs
    df_out = df_feats.copy()
    num_cols = df_out.select_dtypes(include=[np.number]).columns
    df_out[num_cols] = df_out[num_cols].round(3)
    df_out.to_csv("user_segments_and_anomalies.csv", index=False)

    # --- Identify the segment most likely to be bots using a composite z-score ---
    try:
        # Features to use for composite score
        comp_features = {
            'anomaly_score_val': 2.0,   # higher (more positive) means more normal; we want negative -> invert later
            'repeated_ratio': 1.0,
            'link_ratio': 1.0,
            'lexical_diversity': -1.0,  # lower lexical diversity is more bot-like
            'comments_per_day': 1.0,
            'hour_entropy': -1.0,       # lower entropy suggests automation
        }

        # Prepare z-scored columns
        df_comp = df_feats.copy()
        for col in comp_features.keys():
            if col not in df_comp.columns:
                df_comp[col] = 0
        # compute z-scores safely (handle zero std)
        for col in comp_features.keys():
            std = df_comp[col].std(ddof=0)
            mean = df_comp[col].mean()
            if std == 0 or np.isnan(std):
                df_comp[f"z_{col}"] = 0.0
            else:
                df_comp[f"z_{col}"] = (df_comp[col] - mean) / std

        # Note: anomaly_score_val: IsolationForest.decision_function yields higher for normal, lower for anomalous.
        # We will invert its z-score so that higher composite means more bot-like.
        comp_values = []
        for col, weight in comp_features.items():
            zcol = f"z_{col}"
            if col == 'anomaly_score_val':
                comp_values.append((-1.0 * df_comp[zcol]) * weight)
            else:
                comp_values.append(df_comp[zcol] * weight)

        df_comp['composite_score'] = np.sum(np.vstack(comp_values), axis=0)

        # Aggregate per segment
        seg_scores = df_comp.groupby('segment')['composite_score'].mean().reset_index()
        seg_scores = seg_scores.sort_values('composite_score', ascending=False)
        top_seg = int(seg_scores.iloc[0]['segment'])
        print(f"[INFO] Segment most likely bots (composite): {top_seg}")

        users_in_top = df_feats[df_feats['segment'] == top_seg]['author_id'].unique().tolist()
        comments_top_seg = df_comments[df_comments['author_id'].isin(users_in_top)][['author_id', 'content']].copy()
        comments_top_seg = comments_top_seg.rename(columns={'author_id': 'user_id', 'content': 'comment'})

        if comments_top_seg.empty:
            print("[WARN] No comments found for top segment; skipping sample generation.")
        else:
            sample_comments = comments_top_seg.sample(n=min(100, len(comments_top_seg)), random_state=RANDOM_STATE)
            sample_comments.to_csv('most_likely_bots_segment_comments.csv', index=False)
            try:
                df_to_png(sample_comments.head(10), 'most_likely_bots_segment_comments_top10.png')
            except Exception as e:
                print(f"[WARN] Failed to render most_likely_bots_segment_comments_top10.png: {e}")
    except Exception as e:
        print(f"[WARN] Failed to generate top-segment comment samples: {e}")

    # Optional: export tables as PNG images for quick viewing
    if getattr(args, "save_png", False):
        print("[INFO] Rendering PNGs of reports...")
        try:
            df_to_png(segment_report.copy(), "segment_report.png")
        except Exception as e:
            print(f"[WARN] segment_report PNG failed: {e}")

        try:
            df_to_png(anomaly_report.reset_index().copy(), "anomaly_report.png")
        except Exception as e:
            print(f"[WARN] anomaly_report PNG failed: {e}")

    print("[INFO] Done. Outputs:\n  - segment_report.csv\n  - anomaly_report.csv\n  - user_segments_and_anomalies.csv\n  - user_features_v3 (DuckDB table)")
    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bot detection pipeline against a DuckDB dataset.")
    parser.add_argument("--sample", type=int, help="sample number of authors to process (optional)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="authors per parallel chunk")
    parser.add_argument("--max-clusters", type=int, default=MAX_CLUSTERS, help="max clusters to test")
    parser.add_argument("--contamination", type=float, default=0.05, help="IsolationForest contamination (anomaly fraction)")
    parser.add_argument("--workers", type=int, help="number of parallel workers for feature computation")
    parser.add_argument("--year", type=int, help="limit comments to a specific year (uses comments.year column)")
    parser.add_argument("--save-png", action='store_true', help="save PNG images of summary tables (segment_report.png, anomaly_report.png)")
    args = parser.parse_args()
    main(args)
