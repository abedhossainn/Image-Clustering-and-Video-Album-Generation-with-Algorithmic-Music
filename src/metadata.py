import argparse
import csv
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


def _convert_to_degrees(value):
    d = float(value[0][0]) / float(value[0][1])
    m = float(value[1][0]) / float(value[1][1])
    s = float(value[2][0]) / float(value[2][1])
    return d + (m / 60.0) + (s / 3600.0)


def _convert_gps_coordinate(gps_value, ref):
    """Convert GPS coordinate with direction reference."""
    if not gps_value:
        return None
    try:
        degrees = _convert_to_degrees(gps_value)
        if ref in ("S", "W"):
            degrees = -degrees
        return degrees
    except Exception:
        return None


def extract_exif(path: Path):
    try:
        with Image.open(path) as img:
            info = img._getexif() or {}
    except Exception:
        return {}

    result = {}
    gps_info = {}
    for tag_id, value in info.items():
        tag = TAGS.get(tag_id, tag_id)
        if tag == "GPSInfo":
            for key in value.keys():
                subtag = GPSTAGS.get(key, key)
                gps_info[subtag] = value[key]
        else:
            result[tag] = value

    # GPS conversion
    lat = _convert_gps_coordinate(gps_info.get("GPSLatitude"), gps_info.get("GPSLatitudeRef"))
    lon = _convert_gps_coordinate(gps_info.get("GPSLongitude"), gps_info.get("GPSLongitudeRef"))

    return {
        "DateTimeOriginal": result.get("DateTimeOriginal"),
        "GPSLatitude": lat,
        "GPSLongitude": lon,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract minimal EXIF metadata (time/GPS)")
    parser.add_argument("--index", required=True, help="Index CSV with image paths")
    parser.add_argument("--out", required=True, help="Output CSV for metadata")
    args = parser.parse_args()

    rows = []
    idx = Path(args.index)
    with idx.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = Path(row["path"]) if row.get("path") else Path(row["rel_path"])  # fallback
            meta = extract_exif(p)
            meta_row = {
                "path": str(p),
                "DateTimeOriginal": meta.get("DateTimeOriginal"),
                "GPSLatitude": meta.get("GPSLatitude"),
                "GPSLongitude": meta.get("GPSLongitude"),
            }
            rows.append(meta_row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["path","DateTimeOriginal","GPSLatitude","GPSLongitude"]) 
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote metadata for {len(rows)} images -> {out}")

if __name__ == "__main__":
    main()
