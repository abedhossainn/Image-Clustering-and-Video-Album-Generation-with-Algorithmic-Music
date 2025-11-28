import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import open_clip
from tqdm import tqdm


def load_model(model_name: str, device: str):
    if model_name == "clip_vit_b32":
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    elif model_name == "clip_vit_l14":
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="laion2b_s32b_b82k")
    else:
        raise ValueError(f"Unsupported model {model_name}")
    model.eval()
    model.to(device)
    return model, preprocess


def embed_images(index_csv: Path, model_name: str, device: str, batch_size: int, out_npz: Path):
    df = pd.read_csv(index_csv)
    model, preprocess = load_model(model_name, device)

    images = df["path"].tolist()
    embs = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(images), batch_size)):
            batch_paths = images[i:i+batch_size]
            batch_imgs = []
            for p in batch_paths:
                im = Image.open(p).convert("RGB")
                batch_imgs.append(preprocess(im))
            batch = torch.stack(batch_imgs).to(device)
            feat = model.encode_image(batch)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            embs.append(feat.cpu().numpy())
    embs = np.concatenate(embs, axis=0)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, embeddings=embs)
    print(f"Saved embeddings: {embs.shape} -> {out_npz}")


def main():
    parser = argparse.ArgumentParser(description="Extract image embeddings with CLIP")
    parser.add_argument("--input", required=True, help="Index CSV with image paths")
    parser.add_argument("--model", default="clip_vit_b32", choices=["clip_vit_b32", "clip_vit_l14"], help="Embedding model")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--out", required=True, help="Output NPZ path for embeddings")
    args = parser.parse_args()

    embed_images(Path(args.input), args.model, args.device, args.batch_size, Path(args.out))

if __name__ == "__main__":
    main()
