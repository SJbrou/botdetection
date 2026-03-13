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

    anomalies = df_feats[df_feats["anomaly_score"] == -1]
    anomaly_report = anomalies[feature_columns + ["anomaly_score_val"]].describe().transpose()
    # Round anomaly report stats to 3 decimals
    anomaly_report = anomaly_report.round(3)
    anomaly_report.to_csv("anomaly_report.csv")

    # Round numeric columns in the per-user output to 3 decimals for table outputs
    df_out = df_feats.copy()
    num_cols = df_out.select_dtypes(include=[np.number]).columns
    df_out[num_cols] = df_out[num_cols].round(3)
    df_out.to_csv("user_segments_and_anomalies.csv", index=False)

    # Optional: export tables as PNG images for quick viewing
    if getattr(args, "save_png", False):
        def df_to_png(df, filename, row_height=0.25):
            try:
                nrows = max(1, len(df))
                figsize = (10, max(2, nrows * row_height))
                fig, ax = plt.subplots(figsize=figsize)
                ax.axis('off')
                tbl = ax.table(cellText=df.fillna("").values, colLabels=list(df.columns), loc='center')
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(8)
                tbl.scale(1, 1.5)
                plt.tight_layout()
                fig.savefig(filename, dpi=200)
                plt.close(fig)
            except Exception as e:
                print(f"[WARN] Failed to render PNG {filename}: {e}")

        print("[INFO] Rendering PNGs of reports...")
        try:
            df_seg_png = segment_report.copy()
            df_to_png(df_seg_png, "segment_report.png")
        except Exception as e:
            print(f"[WARN] segment_report PNG failed: {e}")

        try:
            df_anom_png = anomaly_report.copy()
            # anomaly_report might be a DataFrame with index as metric names; reset index to show index as column
            df_anom_png = df_anom_png.reset_index()
            df_to_png(df_anom_png, "anomaly_report.png")
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
