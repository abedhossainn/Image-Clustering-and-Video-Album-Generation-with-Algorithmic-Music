import argparse
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

try:
    import umap
except ImportError:
    umap = None


def main():
    parser = argparse.ArgumentParser(description="Reduce embedding dimensionality and optionally plot UMAP")
    parser.add_argument("--embeddings", required=True, help="NPZ file with 'embeddings'")
    parser.add_argument("--pca-dim", type=int, default=64, help="Target PCA dimensions")
    parser.add_argument("--umap", action="store_true", help="Also compute UMAP for visualization")
    parser.add_argument("--out", required=True, help="Output NPZ path for reduced embeddings")
    parser.add_argument("--plot", help="Output PNG path for UMAP plot")
    args = parser.parse_args()

    data = np.load(args.embeddings)
    X = data["embeddings"]

    x_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    pca = PCA(n_components=args.pca_dim, random_state=42)
    x_pca = pca.fit_transform(x_norm)

    out_npz = Path(args.out)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, pca=x_pca)
    print(f"Saved PCA-reduced embeddings: {x_pca.shape} -> {out_npz}")

    if args.umap:
        if umap is None:
            print("UMAP not installed; skipping UMAP plot")
            return
        reducer = umap.UMAP(n_components=2, random_state=42)
        x_umap = reducer.fit_transform(x_pca)
        if args.plot:
            plt.figure(figsize=(6, 5))
            plt.scatter(x_umap[:,0], x_umap[:,1], s=3, alpha=0.6)
            plt.title("UMAP of Embeddings")
            plot_path = Path(args.plot)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=150)
            print(f"Saved UMAP plot -> {plot_path}")

if __name__ == "__main__":
    main()
