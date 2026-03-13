import duckdb
import json
import glob

DUCKDB_PATH = "duckdb_authors.duckdb"
COMMENTS_BASE = "comments"

START_YEAR = 2006
END_YEAR = 2025
BATCH_SIZE = 10_000

con = duckdb.connect(DUCKDB_PATH)
con.execute("PRAGMA threads = 4")

# -------------------------
# RESET TABLES (RAW STAGE)
# -------------------------
con.execute("DROP TABLE IF EXISTS comments_raw")
con.execute("DROP TABLE IF EXISTS comments_raw_years_loaded")

con.execute("""
CREATE TABLE comments_raw (
    comment_id BIGINT PRIMARY KEY,
    post_id BIGINT,
    parent_comment_id BIGINT,
    author_id BIGINT,
    approved BOOLEAN,
    creation_datetime TIMESTAMP,
    content VARCHAR,
    kudos_count BIGINT,
    reference_id BIGINT
);
""")

con.execute("""
CREATE TABLE comments_raw_years_loaded (
    year INTEGER PRIMARY KEY,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
""")

# -------------------------
# Helpers
# -------------------------
def flatten_comment(comment, post_id, parent_id=None):
    yield (
        comment.get("id"),
        post_id,
        parent_id,
        comment.get("author"),
        comment.get("approved"),
        comment.get("creation_datetime"),
        comment.get("content"),
        comment.get("kudos_count"),
        comment.get("reference_id"),
    )

    for child in comment.get("child_comments") or []:
        yield from flatten_comment(child, post_id, comment.get("id"))

# -------------------------
# Load data
# -------------------------
for year in range(START_YEAR, END_YEAR + 1):
    files = glob.glob(f"{COMMENTS_BASE}/{year}/*.json")
    print(f"📥 Loading {len(files)} files for {year}")

    batch = []

    for path in files:
        with open(path, "r") as f:
            data = json.load(f)

        summary = data.get("summary")
        if not summary or "id" not in summary:
            continue

        post_id = summary["id"]
        comments = data.get("comments") or []

        for comment in comments:
            for row in flatten_comment(comment, post_id):
                if row[0] is None:
                    continue

                batch.append(row)

                if len(batch) >= BATCH_SIZE:
                    con.executemany(
                        "INSERT OR IGNORE INTO comments_raw VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        batch,
                    )
                    batch.clear()

    if batch:
        con.executemany(
            "INSERT OR IGNORE INTO comments_raw VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            batch,
        )

    con.execute(
        "INSERT INTO comments_raw_years_loaded (year) VALUES (?)",
        [year]
    )

    print(f"✅ Finished {year}")

con.close()
print("🎉 All comments loaded successfully.")
