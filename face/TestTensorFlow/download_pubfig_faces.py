import pathlib
import random
import requests
import cv2
import numpy as np

# ================== Config ==================
# Max number of NEW face images to download per run
MAX_NEW_IMAGES = 500

# Train / val split ratio, e.g. 0.8 = 80% train, 20% val
TRAIN_RATIO = 0.8

# Timeout for HTTP requests (seconds)
REQUEST_TIMEOUT = 10

# Log file for failed images (so we don't retry dead links)
FAILED_LOG_NAME = "invalid_urls.txt"


# ============================================


def main():
    # Base directory = project root (where this script and dev_urls.txt are located)
    base_dir = pathlib.Path(__file__).resolve().parent

    dev_file = base_dir / "dev_urls.txt"

    train_face_dir = base_dir / "data" / "train" / "face"
    val_face_dir = base_dir / "data" / "val" / "face"

    failed_log_path = base_dir / FAILED_LOG_NAME

    # Make sure target folders exist
    train_face_dir.mkdir(parents=True, exist_ok=True)
    val_face_dir.mkdir(parents=True, exist_ok=True)

    if not dev_file.exists():
        print(f"dev_urls.txt not found at: {dev_file}")
        return

    # Load existing successful files
    existing_files = set()
    for path in list(train_face_dir.glob("*.jpg")) + list(val_face_dir.glob("*.jpg")):
        existing_files.add(path.name)

    # Load previously failed files (dead links etc.)
    failed_files = set()
    if failed_log_path.exists():
        with failed_log_path.open("r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    failed_files.add(name)

    # Union: any file in known_files will be skipped
    known_files = existing_files | failed_files

    print(f"Existing saved faces: {len(existing_files)}")
    print(f"Previously failed faces: {len(failed_files)}")
    print(f"Known filenames total: {len(known_files)}")

    session = requests.Session()
    new_saved_count = 0

    # Open failed log in append mode
    failed_log = failed_log_path.open("a", encoding="utf-8")

    with dev_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.strip().split("\t")
            if len(parts) != 5:
                continue

            person, img_num, url, rect_str, md5sum = parts

            safe_person = person.replace(" ", "_")
            filename = f"{safe_person}_{img_num}.jpg"

            # Skip if we already tried this filename (success or failure)
            if filename in known_files:
                continue

            try:
                resp = session.get(url, timeout=REQUEST_TIMEOUT)
                if resp.status_code != 200:
                    print(f"HTTP {resp.status_code} for url: {url}")
                    failed_files.add(filename)
                    failed_log.write(filename + "\n")
                    continue

                img_array = np.frombuffer(resp.content, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"Failed to decode image from url: {url}")
                    failed_files.add(filename)
                    failed_log.write(filename + "\n")
                    continue

                try:
                    x0, y0, x1, y1 = map(int, rect_str.split(","))
                except ValueError:
                    print(f"Invalid rect '{rect_str}' for url: {url}")
                    failed_files.add(filename)
                    failed_log.write(filename + "\n")
                    continue

                h, w = img.shape[:2]
                x0 = max(0, min(x0, w - 1))
                x1 = max(0, min(x1, w - 1))
                y0 = max(0, min(y0, h - 1))
                y1 = max(0, min(y1, h - 1))

                if x1 <= x0 or y1 <= y0:
                    print(f"Invalid clamped rect for url: {url}")
                    failed_files.add(filename)
                    failed_log.write(filename + "\n")
                    continue

                face = img[y0:y1, x0:x1]
                if face.size == 0:
                    print(f"Empty face crop for url: {url}")
                    failed_files.add(filename)
                    failed_log.write(filename + "\n")
                    continue

                # Decide whether this image goes to train or val
                if random.random() < TRAIN_RATIO:
                    target_dir = train_face_dir
                else:
                    target_dir = val_face_dir

                save_path = target_dir / filename
                cv2.imwrite(str(save_path), face)

                known_files.add(filename)
                new_saved_count += 1
                print(f"[+{new_saved_count}] Saved new: {save_path}")

                if new_saved_count >= MAX_NEW_IMAGES:
                    break

            except Exception as e:
                print(f"Error for url {url}: {e}")
                failed_files.add(filename)
                failed_log.write(filename + "\n")
                continue

    failed_log.close()
    print(f"Finished. Newly downloaded faces this run: {new_saved_count}")


if __name__ == "__main__":
    main()
