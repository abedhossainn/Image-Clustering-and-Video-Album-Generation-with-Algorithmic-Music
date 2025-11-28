import argparse
import subprocess
from pathlib import Path


# Constants for file paths
INDEX_CSV = "data/interim/index.csv"
REDUCED_NPZ = "data/processed/reduced.npz"
LABELS_CSV = "data/processed/labels.csv"


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline")
    parser.add_argument("--root", default="data/raw/butterfly-image-classification", help="Dataset root")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for embedding")
    parser.add_argument("--kmeans", action="store_true", help="Use KMeans instead of HDBSCAN")
    args = parser.parse_args()

    Path("data/interim").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    run(["python", "-m", "src.ingest", "--input", args.root, "--out", INDEX_CSV]) 
    run(["python", "-m", "src.sanity_checks", "--root", args.root, "--report", "reports/sanity.json"]) 
    run(["python", "-m", "src.embed", "--input", INDEX_CSV, "--model", "clip_vit_b32", "--device", args.device, "--batch-size", "32", "--out", "data/interim/embeddings.npz"]) 
    run(["python", "-m", "src.reduce", "--embeddings", "data/interim/embeddings.npz", "--pca-dim", "64", "--umap", "--out", REDUCED_NPZ, "--plot", "reports/umap.png"]) 
    if args.kmeans:
        run(["python", "-m", "src.cluster", "--embeddings", REDUCED_NPZ, "--method", "kmeans", "--k", "60", "--out", LABELS_CSV]) 
    else:
        run(["python", "-m", "src.cluster", "--embeddings", REDUCED_NPZ, "--method", "hdbscan", "--metric", "euclidean", "--min-cluster-size", "20", "--min-samples", "10", "--out", LABELS_CSV]) 
    run(["python", "-m", "src.make_video", "--index", INDEX_CSV, "--clusters", LABELS_CSV, "--resolution", "1920x1080", "--fps", "30", "--duration-per", "3.0", "--crossfade", "0.8", "--engine", "moviepy", "--out", "videos/"]) 

if __name__ == "__main__":
    main()
