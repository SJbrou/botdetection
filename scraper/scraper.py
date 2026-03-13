import requests
import time
import json
import os
import random
from datetime import date, timedelta

# =========================
# Configuration (secrets)
# =========================
# Put real values in `scraper/secrets_local.py` (gitignored).
try:
    from .secrets_local import SECRET_DOMAIN, APP_PARAM, NSFW_HEADER_NAME  # type: ignore
except Exception:
    # Helpful fallback: instruct user to copy the example file
    raise RuntimeError(
        "Missing secrets: create `scraper/secrets_local.py` by copying `scraper/secrets_example.py` and filling in values."
    )

POSTS_BASE_URL = f"https://post.{SECRET_DOMAIN}/api/v1.0/latest/date/{{date}}/"
COMMENTS_BASE_URL = f"https://comment.{SECRET_DOMAIN}/api/v1.0/articles/{{composite_id}}/comments"

START_DATE = date(2019, 10, 1)
END_DATE   = date(2019, 12, 31)

POSTS_DIR = "posts_by_date"
COMMENTS_DIR = "comments"

RANDOMIZE_DATES = False
REFETCH_POSTS_BY_DATE = True   # True = API, False = use stored JSON only

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
# Retry handling (429 only)
# =========================
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 15
BACKOFF_MAX_SECONDS = 300

# =========================
# Setup directories
# =========================
os.makedirs(POSTS_DIR, exist_ok=True)
os.makedirs(COMMENTS_DIR, exist_ok=True)

# =========================
# Helpers
# =========================
def get_with_429_handling(url, params, allow_5xx=False):
    """
    GET request with 429 retry handling.
    If allow_5xx=True, 5xx responses will return None instead of raising.
    """
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)

            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                sleep_time = (
                    int(retry_after)
                    if retry_after and retry_after.isdigit()
                    else min(BACKOFF_BASE_SECONDS * (2 ** attempt), BACKOFF_MAX_SECONDS)
                )
                print(f"429 → sleeping {sleep_time}s")
                time.sleep(sleep_time)
                attempt += 1
                continue

            if r.status_code >= 500:
                if allow_5xx:
                    # return None so caller can skip
                    return None
                r.raise_for_status()

            r.raise_for_status()
            return r

        except requests.exceptions.RequestException as e:
            if allow_5xx and isinstance(e, requests.exceptions.HTTPError) and e.response and e.response.status_code >= 500:
                return None
            raise

    if allow_5xx:
        return None
    raise RuntimeError(f"Max retries exceeded: {url}")


def fetch_posts(api_date: str) -> dict:
    url = POSTS_BASE_URL.format(date=api_date)
    params = {"app": APP_PARAM}
    return get_with_429_handling(url, params).json()


def fetch_comments(composite_id_raw: str) -> dict | None:
    """
    Fetch comments for a post.
    Returns None if API returns 500/502/503 or network error.
    """
    composite_id_api = composite_id_raw.replace("_", "/")
    url = COMMENTS_BASE_URL.format(composite_id=composite_id_api)
    params = {"app": APP_PARAM}

    r = get_with_429_handling(url, params, allow_5xx=True)
    if r is None:
        print(f"      ⚠️ comment API returned 5xx or network error, skipping {composite_id_raw}")
        return None
    return r.json()


def generate_dates(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def comments_complete_for_date(date_str: str, items: list) -> bool:
    for post in items:
        cid = post.get("composite_id")
        if not cid:
            continue

        year = date_str[:4]
        path = os.path.join(
            COMMENTS_DIR,
            year,
            f"{date_str.replace('-', '_')}_{cid}.json"
        )
        if not os.path.exists(path):
            return False
    return True


# =========================
# Main workflow
# =========================
dates = list(generate_dates(START_DATE, END_DATE))
if RANDOMIZE_DATES:
    random.shuffle(dates)

for current_date in dates:
    date_str = current_date.isoformat()
    year = current_date.year
    posts_file = os.path.join(POSTS_DIR, f"api_date_{date_str}.json")

    print(f"\n[DATE] {date_str}")

    # ---- Load or fetch posts ----
    if REFETCH_POSTS_BY_DATE:
        print("  fetching posts from API")
        posts_json = fetch_posts(date_str)
        with open(posts_file, "w", encoding="utf-8") as f:
            json.dump(posts_json, f, ensure_ascii=False, indent=2)
    else:
        if not os.path.exists(posts_file):
            print("  ⚠️ posts file missing, skipping date")
            continue
        print("  loading posts from disk")
        with open(posts_file, "r", encoding="utf-8") as f:
            posts_json = json.load(f)

    items = posts_json.get("items")
    if not items:
        print("  no posts")
        continue

    # ---- Skip date if all comments already collected ----
    if comments_complete_for_date(date_str, items):
        print("  all comments already collected")
        continue

    year_dir = os.path.join(COMMENTS_DIR, str(year))
    os.makedirs(year_dir, exist_ok=True)

    new_comments = 0

    # ---- Fetch missing comments ----
    for post in items:
        cid = post.get("composite_id")
        if not cid:
            continue

        comments_file = os.path.join(
            year_dir,
            f"{date_str.replace('-', '_')}_{cid}.json"
        )

        if os.path.exists(comments_file):
            continue

        print(f"    fetching comments {cid}")

        comments_json = fetch_comments(cid)
        if comments_json is None:
            continue

        with open(comments_file, "w", encoding="utf-8") as f:
            json.dump(comments_json, f, ensure_ascii=False, indent=2)

        new_comments += 1

    print(f"  collected {new_comments} new comment files")

print("\nPosts + comments synchronization complete.")
