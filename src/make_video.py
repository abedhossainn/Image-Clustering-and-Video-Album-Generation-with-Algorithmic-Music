import argparse
import csv
from pathlib import Path
from typing import List
import moviepy


def load_index(index_csv: Path):
    rows = []
    with index_csv.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def images_for_cluster(index_rows, labels_csv: Path, target_label: int):
    import pandas as pd
    labels = pd.read_csv(labels_csv)
    sel = labels[labels["label"] == target_label].index.tolist()
    # index_rows are in the same order as embeddings; we assume this alignment
    images = [index_rows[i]["path"] for i in sel]
    return images


def build_slideshow(image_paths: List[str], duration_per: float, resolution: str, crossfade: float):
    w, h = map(int, resolution.split("x"))
    clips = []
    # Limit to first 30 images for demo video
    limited_paths = image_paths[:30] if len(image_paths) > 30 else image_paths
    for p in limited_paths:
        ic = moviepy.ImageClip(p).with_duration(duration_per)
        ic = ic.resized(height=h)
        comp = moviepy.CompositeVideoClip([ic.with_position('center')], size=(w, h), bg_color=(0, 0, 0)).with_duration(duration_per)
        clips.append(comp)
    # Simple concatenation (no crossfade to keep MoviePy 2.x compatible)
    return moviepy.concatenate_videoclips(clips, method="chain")


def load_music_plan(music_plan_path: Path, cluster_id: int):
    """Load music track for a specific cluster from CSV plan."""
    mp = list(csv.DictReader(open(music_plan_path)))
    for row in mp:
        if int(row["cluster"]) == cluster_id:
            return row["track"]
    return None


def attach_audio_to_video(video, track_path: str):
    """Attach audio clip to video, matching duration with trim/loop and fades."""
    if not track_path or not Path(track_path).exists():
        return video, None
    
    audio_clip = moviepy.AudioFileClip(track_path)
    vdur = video.duration
    
    try:
        if audio_clip.duration >= vdur:
            audio_clip = audio_clip.subclipped(0, vdur)
        else:
            loops = int(vdur // max(audio_clip.duration, 0.1)) + 1
            audio_clip = moviepy.concatenate_audioclips([audio_clip] * loops).subclipped(0, vdur)
        audio_clip = audio_clip.audio_fadein(0.5).audio_fadeout(0.5)
    except Exception:
        pass
    
    return video.with_audio(audio_clip), audio_clip


def render_cluster_video(cluster_id: int, images: list, args, out_dir: Path, music_plan_path=None):
    """Render a single cluster video with optional music."""
    if not images:
        return
    
    video = build_slideshow(images, args.duration_per, args.resolution, args.crossfade)
    
    audio_clip = None
    if music_plan_path:
        track = load_music_plan(music_plan_path, cluster_id)
        video, audio_clip = attach_audio_to_video(video, track)
    
    outfile = out_dir / f"cluster_{cluster_id}.mp4"
    video.write_videofile(str(outfile), fps=args.fps)
    
    if audio_clip:
        audio_clip.close()
    
    print(f"Wrote {outfile}")


def main():
    parser = argparse.ArgumentParser(description="Render slideshow videos per cluster")
    parser.add_argument("--index", required=True, help="Index CSV with image paths")
    parser.add_argument("--clusters", required=True, help="Labels CSV with 'label'")
    parser.add_argument("--music-plan", help="CSV mapping cluster->track (optional)")
    parser.add_argument("--resolution", default="1920x1080", help="Output resolution WxH")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--duration-per", type=float, default=3.0, help="Seconds per image")
    parser.add_argument("--crossfade", type=float, default=0.8, help="Crossfade seconds")
    parser.add_argument("--engine", default="moviepy", help="Rendering engine (reserved)")
    parser.add_argument("--out", required=True, help="Output folder for videos")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_index(Path(args.index))
    import pandas as pd
    labels_df = pd.read_csv(args.clusters)
    cluster_ids = sorted([int(x) for x in set(labels_df["label"]) if int(x) != -1])

    music_plan_path = Path(args.music_plan) if args.music_plan else None

    for cid in cluster_ids:
        imgs = images_for_cluster(rows, Path(args.clusters), cid)
        render_cluster_video(cid, imgs, args, out_dir, music_plan_path)


if __name__ == "__main__":
    main()
