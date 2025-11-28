import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan


def main():
    parser = argparse.ArgumentParser(description="Cluster embeddings with KMeans or HDBSCAN")
    parser.add_argument("--embeddings", required=True, help="NPZ file with PCA embeddings ('pca')")
    parser.add_argument("--method", choices=["kmeans", "hdbscan"], required=True)
    parser.add_argument("--k", type=int, default=60, help="Number of clusters for KMeans")
    parser.add_argument("--min-cluster-size", type=int, default=20, help="HDBSCAN min_cluster_size")
    parser.add_argument("--min-samples", type=int, default=10, help="HDBSCAN min_samples")
    parser.add_argument("--metric", default="euclidean", help="Distance metric (HDBSCAN)")
    parser.add_argument("--out", required=True, help="Output CSV path for labels")
    args = parser.parse_args()

    data = np.load(args.embeddings)
    if "pca" in data:
        X = data["pca"]
    elif "embeddings" in data:
        X = data["embeddings"]
    else:
        raise ValueError("No suitable array found in embeddings NPZ")

    if args.method == "kmeans":
        km = KMeans(n_clusters=args.k, n_init=10, max_iter=300, random_state=42)
        labels = km.fit_predict(X)
        try:
            sil = silhouette_score(X, labels, random_state=42)
        except Exception:
            sil = None
        df = pd.DataFrame({"label": labels})
        df.to_csv(args.out, index=False)
        print(f"KMeans done: k={args.k}, silhouette={sil}. Labels -> {args.out}")
    else:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=args.min_samples, metric=args.metric)
        labels = clusterer.fit_predict(X)
        df = pd.DataFrame({"label": labels})
        df.to_csv(args.out, index=False)
        noise = (labels == -1).sum()
        print(f"HDBSCAN done: clusters={len(set(labels)) - (1 if -1 in labels else 0)}, noise={noise}. Labels -> {args.out}")

if __name__ == "__main__":
    main()
