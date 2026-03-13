import duckdb
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

# =========================
# Configuration
# =========================
POSTS_DIR = Path("posts_by_date")
DATA_DIR = Path("data")
DB_FILE = DATA_DIR / "data.duckdb"

MAX_WORKERS = 7           # threads for JSON parsing
BATCH_SIZE = 2000         # DB insert batch size

LOG_FILE = "load_posts_errors.log"

# =========================
# Setup
# =========================
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(message)s",
)

# =========================
# Connect to DuckDB
# =========================
conn = duckdb.connect(DB_FILE)

# =========================
# Create posts table
# =========================
conn.execute("""
CREATE TABLE IF NOT EXISTS posts (
    id VARCHAR PRIMARY KEY,
    id_int BIGINT,
    composite_id VARCHAR,
    published_at TIMESTAMP,
    upload_id VARCHAR,
    title VARCHAR,
    description VARCHAR,
    tags VARCHAR[],
    nsfw BOOLEAN,
    nopreroll BOOLEAN,
    secret BOOLEAN,
    partner_content BOOLEAN,
    media_type VARCHAR,
    stream_uri VARCHAR,
    kudos_total BIGINT,
    views_total BIGINT,
    stats_id BIGINT,
    created_at TIMESTAMP,
    date TIMESTAMP
);
""")

# =========================
# Resume logic: already loaded post IDs
# =========================
existing_ids = set(
    r[0] for r in conn.execute("SELECT id FROM posts").fetchall()
)

# =========================
# Helpers
# =========================
def get_stream_uri(post):
    if post.get("media_type") != "VIDEO":
        return None
    for media in post.get("media", []):
        for variant in media.get("variants", []):
            if variant.get("version") == "stream":
                return variant.get("uri")
    return None


def parse_json_file(file_path):
    """Parse a single posts_by_date JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logging.warning(f"Failed to parse {file_path}: {e}")
        return []

    items = data.get("items")
    if not items:
        return []

    rows = []
    for post in items:
        post_id = post.get("id")
        if not post_id or post_id in existing_ids:
            continue

        tags = post.get("tags") or ""
        tags_list = tags.split() if tags else []

        stats = post.get("stats", {})

        rows.append((
            post_id,
            post.get("id_int"),
            post.get("composite_id"),
            post.get("published_at"),
            post.get("upload_id"),
            post.get("title"),
            post.get("description"),
            tags_list,
            post.get("nsfw"),
            post.get("nopreroll"),
            post.get("secret"),
            post.get("partner_content"),
            post.get("media_type"),
            get_stream_uri(post),
            stats.get("kudos_total"),
            stats.get("views_total"),
            stats.get("id"),
            post.get("created_at"),
            post.get("date"),
        ))

    return rows


# =========================
# Collect files
# =========================
json_files = sorted(POSTS_DIR.glob("api_date_*.json"))
print(f"Found {len(json_files)} post JSON files")

# =========================
# Multithreaded parsing + batched inserts
# =========================
batch = []
inserted = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(parse_json_file, f): f for f in json_files}

    for future in tqdm(as_completed(futures), total=len(futures), desc="Loading posts"):
        rows = future.result()
        if not rows:
            continue

        for row in rows:
            batch.append(row)
            existing_ids.add(row[0])  # prevent duplicates in same run

            if len(batch) >= BATCH_SIZE:
                conn.executemany("""
                INSERT INTO posts (
                    id, id_int, composite_id, published_at, upload_id,
                    title, description, tags, nsfw, nopreroll, secret,
                    partner_content, media_type, stream_uri,
                    kudos_total, views_total, stats_id,
                    created_at, date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch)
                inserted += len(batch)
                batch.clear()

# Insert remainder
if batch:
    conn.executemany("""
    INSERT INTO posts (
        id, id_int, composite_id, published_at, upload_id,
        title, description, tags, nsfw, nopreroll, secret,
        partner_content, media_type, stream_uri,
        kudos_total, views_total, stats_id,
        created_at, date
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, batch)
    inserted += len(batch)

conn.close()

print(f"✅ Done. Inserted {inserted} new posts.")
print(f"⚠️ Parse errors logged to {LOG_FILE}")
