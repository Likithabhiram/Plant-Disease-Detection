import random
import shutil
from pathlib import Path

from config import RAW_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR, SEED

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {RAW_DIR}\n"
            "Create dataset/raw and place one folder per class inside it."
        )

    random.seed(SEED)
    class_dirs = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())

    if len(class_dirs) < 2:
        raise ValueError("Expected at least two class folders inside dataset/raw.")

    for split_dir in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0

    for class_dir in class_dirs:
        images = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        random.shuffle(images)

        if len(images) < 3:
            print(f"Skipping {class_dir.name}: fewer than 3 images.")
            continue

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        # Ensure every class has at least one image in each split.
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))

        split_map = {
            TRAIN_DIR: images[:n_train],
            VAL_DIR: images[n_train:n_train + n_val],
            TEST_DIR: images[n_train + n_val:],
        }

        for destination, selected_images in split_map.items():
            class_destination = destination / class_dir.name
            class_destination.mkdir(parents=True, exist_ok=True)
            for image_path in selected_images:
                shutil.copy2(image_path, class_destination / image_path.name)

        total_images += n
        print(
            f"{class_dir.name}: train={len(split_map[TRAIN_DIR])}, "
            f"val={len(split_map[VAL_DIR])}, test={len(split_map[TEST_DIR])}"
        )

    print(f"\nFinished splitting {total_images} images.")
    print("Created dataset/train, dataset/val and dataset/test.")


if __name__ == "__main__":
    main()
