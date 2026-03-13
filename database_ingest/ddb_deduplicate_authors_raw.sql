-- Usage --
-- duckdb data/data.duckdb < ddb_deduplicate_authors_raw.sql
-- -------------------------
-- Create canonical authors table
-- -------------------------
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

-- -------------------------
-- Create username history table
-- -------------------------
CREATE TABLE IF NOT EXISTS author_usernames (
    author_id BIGINT,
    username VARCHAR,
    seen_at TIMESTAMP,
    PRIMARY KEY(author_id, username, seen_at)
);

-- -------------------------
-- Merge authors_raw into authors
-- -------------------------
INSERT INTO authors
SELECT
    author_id AS id,
    ANY_VALUE(username) AS username,
    ANY_VALUE(active) AS active,
    ANY_VALUE(newbie) AS newbie,
    BOOL_OR(banned) AS banned,
    BOOL_OR(premium) AS premium,
    ANY_VALUE(registered_at) AS registered_at,
    MIN(seen_at) AS first_seen_at,
    MAX(seen_at) AS last_seen_at
FROM authors_raw
GROUP BY author_id
ON CONFLICT (id) DO UPDATE SET
    username = EXCLUDED.username,
    active = EXCLUDED.active,
    newbie = EXCLUDED.newbie,
    -- Sticky flags
    banned = authors.banned OR EXCLUDED.banned,
    premium = authors.premium OR EXCLUDED.premium,
    -- Preserve earliest registration
    registered_at = LEAST(authors.registered_at, EXCLUDED.registered_at),
    -- Preserve history
    first_seen_at = LEAST(authors.first_seen_at, EXCLUDED.first_seen_at),
    last_seen_at = GREATEST(authors.last_seen_at, EXCLUDED.last_seen_at);

-- -------------------------
-- Insert all usernames into username history
-- -------------------------
INSERT INTO author_usernames
SELECT DISTINCT
    author_id,
    username,
    seen_at
FROM authors_raw
ON CONFLICT DO NOTHING; -- skip duplicates
