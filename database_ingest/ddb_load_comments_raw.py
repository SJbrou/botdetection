import duckdb
import os
import json
import pandas as pd
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIG ---
DB_PATH = "data/data.duckdb"
COMMENTS_DIR = "comments"
MAX_WORKERS = 4  # Adjust based on CPU cores

# --- CONNECT TO DUCKDB ---
con = duckdb.connect(DB_PATH)

# --- CREATE COMMENTS TABLE IF NOT EXISTS ---
con.execute("""
CREATE TABLE IF NOT EXISTS comments (
    comment_id        BIGINT PRIMARY KEY,
    post_id           BIGINT,
    author_id         INTEGER,
    creation_datetime TIMESTAMP,
    content           TEXT,
    kudos_count       INTEGER,
    approved          BOOLEAN,
    reference_id      BIGINT,
    parent_comment_id BIGINT,
    depth             INTEGER,
    year              INTEGER
)
""")

# --- Flatten nested comments recursively ---
def flatten_comments(comments, parent_id=None, depth=0):
    flat = []
    for c in comments:
        children = c.pop("child_comments", [])
        flat.append({**c, "parent_comment_id": parent_id, "depth": depth})
        if children:
            flat.extend(flatten_comments(children, parent_id=c["id"], depth=depth+1))
    return flat

# --- Process a single JSON file ---
def process_file(file_path, year):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        post_id = data["summary"]["id"]
        comments = data.get("comments") or []  # <-- ensure it's a list
        flat_comments = flatten_comments(comments)

        # Add metadata
        for c in flat_comments:
            c["post_id"] = post_id
            c["year"] = year

        if flat_comments:
            df = pd.DataFrame(flat_comments)
            # Ensure all required columns exist
            for col in ["id", "post_id", "author", "creation_datetime", "content", 
                        "kudos_count", "approved", "reference_id", "parent_comment_id", 
                        "depth", "year"]:
                if col not in df.columns:
                    df[col] = None
            df = df[["id", "post_id", "author", "creation_datetime", "content", 
                     "kudos_count", "approved", "reference_id", "parent_comment_id", 
                     "depth", "year"]]
            df.columns = ["comment_id", "post_id", "author_id", "creation_datetime", "content",
                          "kudos_count", "approved", "reference_id", "parent_comment_id", "depth", "year"]
            print(f"[INFO] Processed file: {file_path} ({len(df)} comments)")
            return df
        else:
            print(f"[INFO] Processed file: {file_path} (0 comments)")
    except Exception as e:
        print(f"[ERROR] Failed to process file {file_path}: {e}")
    return None


# --- Collect all files ---
all_files = []
for year_dir in sorted(os.listdir(COMMENTS_DIR)):
    year_path = os.path.join(COMMENTS_DIR, year_dir)
    if not os.path.isdir(year_path) or not year_dir.isdigit():
        continue
    year = int(year_dir)
    files = glob(os.path.join(year_path, "*.json"))
    all_files.extend([(f, year) for f in files])

print(f"[INFO] Total files to process: {len(all_files)}")

# --- Process files in parallel ---
dfs = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_file = {executor.submit(process_file, f, y): f for f, y in all_files}
    for i, future in enumerate(as_completed(future_to_file), start=1):
        df = future.result()
        if df is not None and not df.empty:
            dfs.append(df)
        if i % 50 == 0 or i == len(all_files):
            print(f"[INFO] Completed {i}/{len(all_files)} files")

# --- Bulk insert into DuckDB ---
if dfs:
    final_df = pd.concat(dfs, ignore_index=True)
    con.register("tmp_comments", final_df)
    con.execute("""
        INSERT INTO comments
        SELECT * FROM tmp_comments
    """)
    print(f"[INFO] Inserted {len(final_df)} comments into DuckDB")

con.close()
print("[INFO] All comments ingested successfully!")
