import requests
import json
import os
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# Configuration (secrets)
# =========================
# Put real values in `scraper/secrets_local.py` (gitignored).
try:
    from .secrets_local import SECRET_DOMAIN, APP_PARAM, NSFW_HEADER_NAME  # type: ignore
except Exception:
    raise RuntimeError(
        "Missing secrets: create `scraper/secrets_local.py` by copying `scraper/secrets_example.py` and filling in values."
    )

POSTS_BASE_URL = f"https://post.{SECRET_DOMAIN}/api/v1.0/latest/date/{{date}}/"
COMMENTS_BASE_URL = f"https://comment.{SECRET_DOMAIN}/api/v1.0/articles/{{composite_id}}/comments"

START_YEAR = 2007
END_YEAR   = 2025

THREADS = 4

POSTS_DIR = "posts_by_date"
COMMENTS_DIR = "comments"

# =========================
# Headers (NSFW enabled)
# =========================
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 15_7_3) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15"
    ),
    "Accept": "application/json",
    "Origin": "https://www.dumpert.nl",
    "Referer": "https://www.dumpert.nl/",
    "x-dumpert-nsfw": "1",
}

# =========================
# Setup directories
# =========================
os.makedirs(POSTS_DIR, exist_ok=True)
os.makedirs(COMMENTS_DIR, exist_ok=True)

# =========================
# HTTP helpers
# =========================
def get(url, params):
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r

def fetch_posts(api_date: str) -> dict:
    url = POSTS_BASE_URL.format(date=api_date)
    return get(url, {"app": APP_PARAM}).json()

def fetch_comments(composite_id_raw: str) -> dict | None:
    composite_id_api = composite_id_raw.replace("_", "/")
    url = COMMENTS_BASE_URL.format(composite_id=composite_id_api)

    try:
        return get(url, {"app": APP_PARAM}).json()
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else None
        if status in (500, 502, 503):
            print(f"      ⚠️ comment API {status}, skipping {composite_id_raw}")
            return None
        raise

# =========================
# Utilities
# =========================
def generate_dates_for_year(year: int):
    d = date(year, 1, 1)
    end = date(year, 12, 31)
    while d <= end:
        yield d
        d += timedelta(days=1)

def comments_complete_for_date(date_str: str, items: list) -> bool:
    year = date_str[:4]
    for post in items:
        cid = post.get("composite_id")
        if not cid:
            continue
        path = os.path.join(
            COMMENTS_DIR,
            year,
            f"{date_str.replace('-', '_')}_{cid}.json"
        )
        if not os.path.exists(path):
            return False
    return True

# =========================
# Worker (one year)
# =========================
def process_year(year: int):
    print(f"\n🧵 START year {year}")
    year_dir = os.path.join(COMMENTS_DIR, str(year))
    os.makedirs(year_dir, exist_ok=True)

    for d in generate_dates_for_year(year):
        date_str = d.isoformat()
        print(f"[{year}] {date_str}")

        try:
            posts_json = fetch_posts(date_str)
        except Exception as e:
            print(f"[{year}] ❌ posts failed {date_str}: {e}")
            continue

        # overwrite posts_by_date
        posts_file = os.path.join(POSTS_DIR, f"api_date_{date_str}.json")
        with open(posts_file, "w", encoding="utf-8") as f:
            json.dump(posts_json, f, ensure_ascii=False, indent=2)

        items = posts_json.get("items")
        if not items:
            continue

        if comments_complete_for_date(date_str, items):
            continue

        for post in items:
            cid = post.get("composite_id")
            if not cid:
                continue

            comment_path = os.path.join(
                year_dir,
                f"{date_str.replace('-', '_')}_{cid}.json"
            )

            if os.path.exists(comment_path):
                continue

            comments = fetch_comments(cid)
            if comments is None:
                continue

            with open(comment_path, "w", encoding="utf-8") as f:
                json.dump(comments, f, ensure_ascii=False, indent=2)

    print(f"✅ DONE year {year}")

# =========================
# Main (thread pool)
# =========================
years = list(range(START_YEAR, END_YEAR + 1))

with ThreadPoolExecutor(max_workers=THREADS) as executor:
    futures = [executor.submit(process_year, y) for y in years]
    for f in as_completed(futures):
        f.result()

print("\n🚀 All years processed.")