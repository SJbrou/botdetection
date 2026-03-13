import duckdb
from pathlib import Path

DUCKDB_PATH = "data/data.duckdb"
COMMENTS_ROOT = Path("comments")

con = duckdb.connect(DUCKDB_PATH)
con.execute("PRAGMA threads = 1")

# Create / ensure schema
con.execute("""
CREATE TABLE IF NOT EXISTS authors (
    id BIGINT PRIMARY KEY,
    username VARCHAR,
    active BOOLEAN,
    newbie BOOLEAN,
    banned BOOLEAN,
    premium BOOLEAN,
    registered_at TIMESTAMP,
    first_seen_at TIMESTAMP,
    last_seen_at TIMESTAMP
);
""")

def load_authors_for_year(year: int):
    json_glob = str(COMMENTS_ROOT / str(year) / "**/*.json")
    print(f"Loading authors from {json_glob}")

    con.execute(f"""
    WITH raw AS (
        SELECT
            author.id,
            author.username,
            author.active,
            author.newbie,
            author.banned,
            author.premium,
            CAST(author.registered_at AS TIMESTAMP) AS registered_at,

            CAST(
                STRPTIME(
                    REGEXP_EXTRACT(filename, '(\\d{{4}}_\\d{{2}}_\\d{{2}})'),
                    '%Y_%m_%d'
                ) AS TIMESTAMP
            ) AS seen_at
        FROM read_json_auto('{json_glob}', filename=true) f
        CROSS JOIN UNNEST(f.authors) AS a(author)
    ),

    aggregated AS (
        SELECT
            id,
            ANY_VALUE(username) AS username,
            ANY_VALUE(active) AS active,
            ANY_VALUE(newbie) AS newbie,
            BOOL_OR(banned) AS banned,
            BOOL_OR(premium) AS premium,
            ANY_VALUE(registered_at) AS registered_at,
            MIN(seen_at) AS first_seen_at,
            MAX(seen_at) AS last_seen_at
        FROM raw
        GROUP BY id
    )

    INSERT INTO authors
    SELECT * FROM aggregated
    ON CONFLICT (id) DO UPDATE SET
        username = EXCLUDED.username,
        active = EXCLUDED.active,
        newbie = EXCLUDED.newbie,

        banned = authors.banned OR EXCLUDED.banned,
        premium = authors.premium OR EXCLUDED.premium,

        registered_at = COALESCE(authors.registered_at, EXCLUDED.registered_at),

        first_seen_at = LEAST(authors.first_seen_at, EXCLUDED.first_seen_at),
        last_seen_at = GREATEST(authors.last_seen_at, EXCLUDED.last_seen_at)
    """)

# Load newest → oldest (recommended)
for year in range(2025, 2006, -1):
    load_authors_for_year(year)

con.close()
