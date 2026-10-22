
import time
import pandas as pd

from google_play_scraper import reviews, app as get_app, Sort

from config import (
    APPS,
    RAW_FILE,
    REVIEWS_PER_APP,
    BATCH_SIZE,
    LANGUAGE,
    COUNTRY,
    REQUEST_PAUSE_SECONDS,
    MAX_ERRORS,
)


def scrape_app(app_name, app_id, target):
    """Collect up to target unique reviews for one app."""

    collected = []
    seen_ids = set()
    continuation_token = None
    errors = 0

    while len(collected) < target:
        batch_size = min(BATCH_SIZE, target - len(collected))

        try:
            batch, next_token = reviews(
                app_id,
                lang=LANGUAGE,
                country=COUNTRY,
                sort=Sort.NEWEST,
                count=batch_size,
                continuation_token=continuation_token,
            )
            errors = 0

        except Exception as exc:
            errors += 1
            print(
                f"{app_name}: error {errors}/{MAX_ERRORS}: {exc}"
            )

            if errors >= MAX_ERRORS:
                print(f"{app_name}: stopping after repeated errors.")
                break

            time.sleep(5)
            continue

        if not batch:
            print(f"{app_name}: no more reviews available.")
            break

        new_count = 0

        for review in batch:
            review_id = review.get("reviewId")

            # Use review ID where available; otherwise use review text.
            if review_id:
                unique_key = str(review_id)
            else:
                unique_key = str(
                    review.get("content", "")
                ).strip().lower()

            if not unique_key or unique_key in seen_ids:
                continue

            seen_ids.add(unique_key)
            collected.append(review)
            new_count += 1

            if len(collected) >= target:
                break

        print(
            f"{app_name}: {len(collected)}/{target} reviews collected"
        )

        if len(collected) >= target or next_token is None:
            break

        if new_count == 0 and next_token == continuation_token:
            print(f"{app_name}: pagination stopped.")
            break

        continuation_token = next_token
        time.sleep(REQUEST_PAUSE_SECONDS)

    if not collected:
        return pd.DataFrame()

    df = pd.DataFrame(collected)

    column_mapping = {
        "content": "review_text",
        "score": "rating",
        "at": "review_date",
        "thumbsUpCount": "thumbs_up",
        "appVersion": "app_version",
        "reviewId": "review_id_original",
    }

    df = df.rename(columns=column_mapping)
    df["app"] = app_name
    df["app_id"] = app_id

    required_columns = [
        "app",
        "app_id",
        "review_id_original",
        "review_text",
        "rating",
        "review_date",
        "thumbs_up",
        "app_version",
    ]

    for column in required_columns:
        if column not in df.columns:
            df[column] = None

    return df[required_columns].head(target)


def main():
    if RAW_FILE.exists():
        answer = input(
            f"{RAW_FILE} already exists. "
            "Collect reviews again? (y/n): "
        ).strip().lower()

        if answer != "y":
            print("Keeping the existing raw review file.")
            return

    frames = []

    for app_name, app_id in APPS.items():
        print(f"\nChecking Google Play listing: {app_name}")

        try:
            metadata = get_app(
                app_id,
                lang=LANGUAGE,
                country=COUNTRY,
            )

            print(
                f"Listing: {metadata.get('title', 'Unknown')} | "
                f"Store rating: {metadata.get('score', 'N/A')}"
            )

        except Exception as exc:
            print(f"Could not verify {app_name}: {exc}")
            continue

        print(f"Collecting reviews for {app_name}...")

        frame = scrape_app(
            app_name,
            app_id,
            REVIEWS_PER_APP,
        )

        if not frame.empty:
            frames.append(frame)
        else:
            print(f"No reviews collected for {app_name}.")

    if not frames:
        print("No reviews collected. Check the app IDs and connection.")
        return

    raw = pd.concat(frames, ignore_index=True)

    # Remove duplicate review IDs within each app.
    raw = raw.drop_duplicates(
        subset=["app", "review_id_original"],
        keep="first",
    )

    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)

    raw.to_csv(
        RAW_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nSaved {len(raw):,} reviews to {RAW_FILE}")
    print("\nReview count by app:")
    print(raw.groupby("app").size())
    print("\nMissing values:")
    print(raw.isna().sum())


if __name__ == "__main__":
    main()
