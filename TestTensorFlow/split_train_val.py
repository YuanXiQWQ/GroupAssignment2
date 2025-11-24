import pathlib
import random
import shutil

# Ratio of images to move from train to val
VAL_RATIO = 0.2  # 20%

def move_subset(src_dir: pathlib.Path, dst_dir: pathlib.Path):
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Collect jpg and png files
    files = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png"))
    if not files:
        print(f"No images found in {src_dir}")
        return

    random.shuffle(files)
    val_count = int(len(files) * VAL_RATIO)
    val_files = files[:val_count]

    print(f"{src_dir} -> {dst_dir}, moving {val_count} / {len(files)} images")

    for path in val_files:
        target = dst_dir / path.name
        shutil.move(str(path), str(target))

def main():
    base_dir = pathlib.Path(__file__).resolve().parent
    train_face = base_dir / "data" / "train" / "face"
    train_non_face = base_dir / "data" / "train" / "non_face"
    val_face = base_dir / "data" / "val" / "face"
    val_non_face = base_dir / "data" / "val" / "non_face"

    move_subset(train_face, val_face)
    move_subset(train_non_face, val_non_face)

if __name__ == "__main__":
    main()
