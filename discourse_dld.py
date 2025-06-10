import requests
import os
import json
from datetime import datetime, timezone # Ensure timezone is imported
from urllib.parse import urljoin, urlencode

# ========== CONFIGURATION ==========

DISCOURSE_BASE_URL = "https://discourse.onlinedegree.iitm.ac.in/"
CATEGORY_SLUG = "courses/tds-kb"
CATEGORY_ID = 34
START_DATE = "2025-01-01" # Inclusive
END_DATE = "2025-04-15"   # Inclusive

RAW_COOKIE_STRING = """""" # Replace with your actual cookie string

OUTPUT_DIR = "discourse_json"
POST_ID_BATCH_SIZE = 50
MAX_CONSECUTIVE_PAGES_WITHOUT_NEW_TOPICS = 5 # New configuration for breaking loop

# ====================================

def parse_cookie_string(raw_cookie_string):
    """Parses a raw cookie string into a dictionary."""
    cookies = {}
    if not raw_cookie_string.strip():
        print("Warning: RAW_COOKIE_STRING is empty. Requests might fail if authentication is needed.")
        return cookies
    for cookie_part in raw_cookie_string.strip().split(";"):
        if "=" in cookie_part:
            key, value = cookie_part.strip().split("=", 1)
            cookies[key] = value
    return cookies


def get_topic_ids(base_url, category_slug, category_id, start_date_str, end_date_str, cookies):
    """Fetches topic IDs from a specific category within a date range."""
    url = urljoin(base_url, f"c/{category_slug}/{category_id}.json")
    topic_ids = []
    page = 0

    start_dt_naive = datetime.fromisoformat(start_date_str + "T00:00:00")
    start_dt = start_dt_naive.replace(tzinfo=timezone.utc)
    end_dt_naive = datetime.fromisoformat(end_date_str + "T23:59:59.999999")
    end_dt = end_dt_naive.replace(tzinfo=timezone.utc)

    print(f"Fetching topic IDs from category between {start_dt} and {end_dt}...")

    # Variables for the new loop break condition
    consecutive_pages_with_no_new_unique_topics = 0
    last_known_unique_topic_count = 0

    while True:
        paginated_url = f"{url}?page={page}"
        try:
            response = requests.get(paginated_url, cookies=cookies, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch page {page}: {e}")
            break

        try:
            data = response.json()
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from page {page}. Content: {response.text[:200]}...")
            break

        topics_on_page = data.get("topic_list", {}).get("topics", [])

        if not topics_on_page:
            print(f"No more topics found on page {page} (API returned empty list).")
            break # Primary stop condition: API says no more topics on this page

        # Store current number of unique topics before processing this page
        # This helps check if *this specific page fetch* added anything new
        count_before_processing_page = len(set(topic_ids))

        for topic in topics_on_page:
            created_at_str = topic.get("created_at")
            if created_at_str:
                try:
                    created_date = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                except ValueError:
                    print(f"Warning: Could not parse date '{created_at_str}' for topic ID {topic.get('id')}")
                    continue

                if start_dt <= created_date <= end_dt:
                    topic_ids.append(topic["id"]) # Add ID, will be deduped later for count

        current_unique_topic_count = len(set(topic_ids))

        if topics_on_page and current_unique_topic_count == count_before_processing_page :
            # This means the current page had topics, but none of them were new *and* within the date range,
            # or all topics fetched from this page were duplicates of ones already in topic_ids from *previous pages*.
            # For the staleness check, we care if the overall unique set isn't growing.
             pass # Handled by the check below using last_known_unique_topic_count

        # Staleness check: Has the *total* number of unique topics found stopped growing?
        if current_unique_topic_count == last_known_unique_topic_count and topics_on_page:
            # topics_on_page is checked to ensure we don't increment if an empty page was returned (which is a valid end)
            consecutive_pages_with_no_new_unique_topics += 1
            print(f"Page {page} did not yield any new unique topics. Consecutive stale pages: {consecutive_pages_with_no_new_unique_topics}.")
        else:
            consecutive_pages_with_no_new_unique_topics = 0 # Reset if new unique topics were found

        last_known_unique_topic_count = current_unique_topic_count

        if consecutive_pages_with_no_new_unique_topics >= MAX_CONSECUTIVE_PAGES_WITHOUT_NEW_TOPICS:
            print(f"No new unique topics found for {MAX_CONSECUTIVE_PAGES_WITHOUT_NEW_TOPICS} consecutive pages. Assuming end of relevant category listing.")
            break

        # Original secondary stop condition (heuristic)
        more_topics_url = data.get("topic_list", {}).get("more_topics_url")
        if not more_topics_url:
            # This typically means it's the last page.
            # The condition `len(topics_on_page) < 30` was a heuristic for when more_topics_url might be missing
            # but the page wasn't full. If more_topics_url is definitively gone, it's a strong signal.
            print(f"No 'more_topics_url' indicated on page {page}. Assuming this is the last page of topics.")
            break
        
        print(f"Fetched page {page}, {len(topics_on_page)} topics on page. Total unique topics found so far: {current_unique_topic_count}. Continuing...")
        page += 1


    final_unique_topic_ids = list(set(topic_ids)) # Deduplicate
    print(f"Total unique topics found in timeframe: {len(final_unique_topic_ids)}")
    return final_unique_topic_ids


def get_full_topic_json(base_url, topic_id, cookies):
    """Fetches the full topic JSON, including all posts by handling pagination."""
    initial_topic_url = urljoin(base_url, f"t/{topic_id}.json")
    print(f"Fetching initial data for topic {topic_id} from {initial_topic_url}")

    try:
        response = requests.get(initial_topic_url, cookies=cookies, timeout=30)
        response.raise_for_status()
        topic_data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch initial topic data for {topic_id}: {e}")
        return None
    except json.JSONDecodeError:
        print(f"Failed to decode initial JSON for topic {topic_id}. Content: {response.text[:200]}...")
        return None

    post_stream = topic_data.get("post_stream")
    if not post_stream or "stream" not in post_stream or "posts" not in post_stream:
        print(f"Error: 'post_stream' not found or incomplete in topic {topic_id}. Skipping post fetching.")
        return topic_data

    all_post_ids_in_stream = post_stream.get("stream", [])
    loaded_post_ids = {post["id"] for post in post_stream.get("posts", [])}

    all_post_ids_in_stream = [pid for pid in all_post_ids_in_stream if pid is not None]

    missing_post_ids = [pid for pid in all_post_ids_in_stream if pid not in loaded_post_ids]

    print(f"Topic {topic_id}: Total posts in stream: {len(all_post_ids_in_stream)}, Initially loaded: {len(loaded_post_ids)}, Missing: {len(missing_post_ids)}")

    if not missing_post_ids:
        print(f"All posts for topic {topic_id} already loaded in initial fetch.")
        return topic_data

    fetched_additional_posts = []
    for i in range(0, len(missing_post_ids), POST_ID_BATCH_SIZE):
        batch_ids = missing_post_ids[i:i + POST_ID_BATCH_SIZE]

        query_params = [("post_ids[]", pid) for pid in batch_ids]
        posts_url = urljoin(base_url, f"t/{topic_id}/posts.json")

        print(f"Fetching batch of {len(batch_ids)} posts for topic {topic_id} (IDs: {batch_ids[0]}...{batch_ids[-1]})")

        try:
            batch_response = requests.get(posts_url, params=query_params, cookies=cookies, timeout=60)
            batch_response.raise_for_status()
            batch_data = batch_response.json()

            if isinstance(batch_data, list):
                 fetched_additional_posts.extend(batch_data)
            elif "post_stream" in batch_data and "posts" in batch_data["post_stream"]:
                fetched_additional_posts.extend(batch_data["post_stream"]["posts"])
            elif "posts" in batch_data and isinstance(batch_data["posts"], list):
                 fetched_additional_posts.extend(batch_data["posts"])
            else:
                print(f"Warning: Unexpected JSON structure for post batch in topic {topic_id}. Data: {str(batch_data)[:200]}...")

        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch post batch for topic {topic_id} (IDs: {batch_ids}): {e}")
        except json.JSONDecodeError:
            print(f"Failed to decode JSON for post batch in topic {topic_id}. Response: {batch_response.text[:200]}...")

    if fetched_additional_posts:
        print(f"Successfully fetched {len(fetched_additional_posts)} additional posts for topic {topic_id}.")
        existing_posts_in_topic_data = {post['id']: post for post in topic_data["post_stream"]["posts"]}
        for post in fetched_additional_posts:
            if post['id'] not in existing_posts_in_topic_data:
                topic_data["post_stream"]["posts"].append(post)
                existing_posts_in_topic_data[post['id']] = post

        post_id_to_post_map = {post['id']: post for post in topic_data["post_stream"]["posts"]}

        sorted_posts = []
        for post_id_val in all_post_ids_in_stream: # Renamed post_id to post_id_val to avoid conflict
            if post_id_val in post_id_to_post_map:
                sorted_posts.append(post_id_to_post_map[post_id_val])

        topic_data["post_stream"]["posts"] = sorted_posts
        print(f"Topic {topic_id}: Final post count in JSON: {len(topic_data['post_stream']['posts'])}")

    return topic_data


def save_topic_json(topic_id, json_data, output_dir):
    """Saves the topic JSON data to a file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"topic_{topic_id}.json")
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        # print(f"Successfully saved topic {topic_id} to {filepath}") # Reduced verbosity
    except IOError as e:
        print(f"Error saving topic {topic_id} to {filepath}: {e}")


def main():
    """Main function to orchestrate the downloading process."""
    print("Script started.")
    cookies = parse_cookie_string(RAW_COOKIE_STRING)
    if not cookies and DISCOURSE_BASE_URL != "https://meta.discourse.org/":
        print("Warning: Running without cookies. This may fail for private forums or specific content.")

    topic_ids = get_topic_ids(
        DISCOURSE_BASE_URL,
        CATEGORY_SLUG,
        CATEGORY_ID,
        START_DATE,
        END_DATE,
        cookies
    )

    if not topic_ids:
        print("No topic IDs found for the given criteria. Exiting.")
        return

    total_topics = len(topic_ids)
    success_downloads = 0
    failed_topic_ids = []

    print(f"\nStarting download of {total_topics} topics...\n")

    for i, topic_id in enumerate(topic_ids, 1):
        print(f"--- [{i}/{total_topics}] Processing topic ID: {topic_id} ---")
        topic_json_data = get_full_topic_json(DISCOURSE_BASE_URL, topic_id, cookies)
        if topic_json_data:
            save_topic_json(topic_id, topic_json_data, OUTPUT_DIR)
            success_downloads += 1
        else:
            print(f"Failed to get complete data for topic {topic_id}.")
            failed_topic_ids.append(topic_id)
        # print(f"--- Finished processing topic ID: {topic_id} ---\n") # Reduced verbosity

    print("\n========= SUMMARY =========")
    print(f"Total topics identified: {total_topics}")
    print(f"Successfully downloaded full data for: {success_downloads} topics")
    print(f"Failed to download/process: {len(failed_topic_ids)} topics")
    if failed_topic_ids:
        print("Failed topic IDs:", failed_topic_ids)
    print(f"Downloaded files are in: {os.path.abspath(OUTPUT_DIR)}")
    print("Script finished.")

if __name__ == "__main__":
    main()