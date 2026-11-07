
"""Legacy batch downloader. Prefer `python -m src.prep.pipeline download`."""

from huggingface_hub import HfApi, hf_hub_download
from pathlib import Path
import subprocess
import sys
import shutil

REPO_ID = "cschell/boxrr-23"
REPO_TYPE = "dataset"

LOCAL_DIR = Path("data/raw")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 200
START = 705

MIN_FREE_GB = 10
MIN_FREE_BYTES = MIN_FREE_GB * 1024**3

INSPECT_SCRIPT = Path("src/download/inspect_multiple_sessions.py")

api = HfApi()

def get_free_space():
    return shutil.disk_usage(LOCAL_DIR).free

repo_files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE
)

archives = sorted([
    file_path
    for file_path in repo_files
    if file_path.startswith("users/") and file_path.endswith(".tar")
])

total_archives = len(archives)

print(f"Total TAR files: {total_archives}")

total_downloaded = 0
batch_number = 1

while START < total_archives:

    selected_archives = archives[START:START + BATCH_SIZE]

    if not selected_archives:
        break

    print("\n" + "=" * 60)
    print(f"=== BATCH {batch_number} ===")
    print(f"Archive range: {START + 1} - {START + len(selected_archives)}")
    print("=" * 60)

    downloaded_count = 0

    for index, archive_path in enumerate(selected_archives, start=1):

        print(f"\nDownloading {index}/{len(selected_archives)}")
        print("Archive:", archive_path)

        local_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=archive_path,
            local_dir=LOCAL_DIR
        )

        downloaded_count += 1
        total_downloaded += 1

        print("Downloaded:", local_path)

    if downloaded_count > 0:
        print("\n=== START SESSION FILTERING ===")

        subprocess.run(
            [sys.executable, str(INSPECT_SCRIPT)],
            check=True
        )

        print("\n=== FILTERING COMPLETE ===")

    START += downloaded_count
    batch_number += 1

    free_bytes = get_free_space()
    free_gb = free_bytes / 1024**3

    print(f"\nNext START: {START}")
    print(f"Total downloaded: {total_downloaded}")
    print(f"Remaining space after filtering: {free_gb:.2f} GB")

    if free_bytes <= MIN_FREE_BYTES:
        print("\nSTOP: Free disk space reached 10 GB after filtering.")
        break

    if downloaded_count < len(selected_archives):
        print("Batch stopped before completion.")
        break

print("\n" + "=" * 60)
print("=== FINAL SUMMARY ===")
print("Total downloaded:", total_downloaded)
print("Next START:", START)
print(f"Remaining disk space: {get_free_space() / 1024**3:.2f} GB")
print("=== COMPLETE ===")
