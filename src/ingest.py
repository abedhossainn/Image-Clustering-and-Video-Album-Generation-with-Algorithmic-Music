import argparse
import os
import sys
import csv
from pathlib import Path

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

def walk_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT:
            yield p

def main():
    parser = argparse.ArgumentParser(description="Index images under a root directory")
    parser.add_argument("--input", required=True, help="Root folder of images")
    parser.add_argument("--out", required=True, help="Output CSV path for index")
    args = parser.parse_args()

    root = Path(args.input)
    if not root.exists():
        print(f"Input path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for img_path in walk_images(root):
        rel = img_path.relative_to(root)
        label = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        rows.append({"path": str(img_path), "rel_path": str(rel), "label": label})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "rel_path", "label"]) 
        writer.writeheader()
        writer.writerows(rows)
    print(f"Indexed {len(rows)} images -> {out_path}")

if __name__ == "__main__":
    main()
