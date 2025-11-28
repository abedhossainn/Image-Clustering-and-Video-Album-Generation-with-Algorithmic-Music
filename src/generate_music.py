"""
Generate music algorithmically from cluster features.
Maps:
  - Cluster size → number of instrument layers
  - Cluster density/variance → tempo and melody complexity
  - Visual features (brightness, saturation) → scale choice and dynamics
"""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from midiutil import MIDIFile
import cv2


def image_features(img_path: str):
    """Extract brightness and saturation from an image."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        brightness = float(np.mean(v) / 255.0)
        saturation = float(np.mean(s) / 255.0)
        return {"brightness": brightness, "saturation": saturation}
    except Exception:
        return None


def compute_cluster_features(index_csv: Path, labels_csv: Path, embeddings_npz: Path):
    """Compute size, variance, and visual attributes per cluster."""
    idx = pd.read_csv(index_csv)
    labels = pd.read_csv(labels_csv)
    data = np.load(embeddings_npz)
    if "pca" in data:
        X = data["pca"]
    elif "embeddings" in data:
        X = data["embeddings"]
    else:
        raise ValueError("No embeddings found in NPZ")
    
    idx = idx.reset_index(drop=True)
    labels = labels.reset_index(drop=True)
    assert len(idx) == len(labels) == len(X), "Length mismatch"
    
    cluster_feats = {}
    for lab in sorted(set(labels["label"])):
        if lab == -1:
            continue
        mask = labels["label"] == lab
        cluster_size = int(mask.sum())
        cluster_embs = X[mask]
        variance = float(np.mean(np.var(cluster_embs, axis=0)))
        
        # Visual features from sample images
        paths = idx.loc[mask, "path"].tolist()[:20]
        vis = [image_features(p) for p in paths]
        vis = [v for v in vis if v is not None]
        brightness = float(np.mean([v["brightness"] for v in vis])) if vis else 0.5
        saturation = float(np.mean([v["saturation"] for v in vis])) if vis else 0.5
        
        cluster_feats[int(lab)] = {
            "size": cluster_size,
            "variance": variance,
            "brightness": brightness,
            "saturation": saturation,
        }
    return cluster_feats


def generate_midi_for_cluster(features: dict, duration_sec: float, out_path: Path):
    """Generate a MIDI file based on cluster features."""
    size = features["size"]
    variance = features["variance"]
    brightness = features["brightness"]
    saturation = features["saturation"]
    
    # Map features to musical parameters
    # Tempo: 80-160 BPM based on saturation and variance
    tempo = int(80 + 50 * saturation + 30 * min(variance / 0.1, 1.0))
    tempo = np.clip(tempo, 80, 160)
    
    # Number of tracks: 1-4 based on size buckets
    num_tracks = int(1 + min(size // 100, 3))
    
    # Melody complexity: note density and pitch range
    note_density = 0.3 + 0.5 * min(variance / 0.1, 1.0)  # notes per beat
    pitch_range = int(12 + 12 * min(variance / 0.1, 1.0))  # semitones
    
    # Scale: major if bright, minor if dark
    if brightness > 0.5:
        scale = [0, 2, 4, 5, 7, 9, 11]  # Major scale
        base_pitch = 60  # C4
    else:
        scale = [0, 2, 3, 5, 7, 8, 10]  # Natural minor
        base_pitch = 57  # A3
    
    # Create MIDI file with random generator
    rng = np.random.default_rng(seed=42)
    midi = MIDIFile(num_tracks)
    for track in range(num_tracks):
        midi.addTempo(track, 0, tempo)
    
    # Duration in beats
    beats_per_second = tempo / 60.0
    total_beats = int(duration_sec * beats_per_second)
    
    # Generate notes for each track
    for track in range(num_tracks):
        # Different instrument/pattern per track
        if track == 0:
            # Melody track
            channel = 0
            program = 0  # Acoustic Grand Piano
            velocity = int(70 + 25 * saturation)
        elif track == 1:
            # Harmony track
            channel = 1
            program = 48  # String Ensemble
            velocity = int(50 + 20 * brightness)
        elif track == 2:
            # Bass track
            channel = 2
            program = 32  # Acoustic Bass
            velocity = 60
            base_pitch -= 24
        else:
            # Pad track
            channel = 3
            program = 88  # Pad (new age)
            velocity = 40
        
        midi.addProgramChange(track, channel, 0, program)
        
        # Generate note sequence
        time = 0.0
        while time < total_beats:
            # Random walk melody within scale
            degree = rng.integers(0, len(scale))
            octave_offset = rng.integers(-pitch_range // 12, pitch_range // 12 + 1) * 12
            pitch = base_pitch + scale[degree] + octave_offset
            pitch = np.clip(pitch, 21, 108)
            
            # Note duration: 0.5-2 beats
            dur = 0.5 + rng.random() * 1.5
            
            midi.addNote(track, channel, pitch, time, dur, velocity)
            
            # Advance time based on note density
            step = 1.0 / note_density if note_density > 0 else 1.0
            time += step * (0.8 + 0.4 * rng.random())
    
    # Write MIDI file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        midi.writeFile(f)
    print(f"Generated MIDI: {out_path}")


def midi_to_wav(midi_path: Path, wav_path: Path, soundfont: str = None):
    """Convert MIDI to WAV using FluidSynth."""
    if soundfont is None:
        # Try common soundfont locations
        candidates = [
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
            "/usr/share/soundfonts/FluidR3_GM.sf2",
            "/usr/share/sounds/sf2/default.sf2",
        ]
        for sf in candidates:
            if Path(sf).exists():
                soundfont = sf
                break
    
    if soundfont is None or not Path(soundfont).exists():
        print("Warning: No soundfont found. Install FluidR3_GM.sf2 or specify --soundfont")
        return False
    
    cmd = [
        "fluidsynth",
        "-ni",
        soundfont,
        str(midi_path),
        "-F", str(wav_path),
        "-r", "44100",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Converted to WAV: {wav_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FluidSynth error: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print("FluidSynth not found. Install with: sudo apt-get install fluidsynth")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate music from cluster features")
    parser.add_argument("--index", required=True, help="Index CSV with image paths")
    parser.add_argument("--clusters", required=True, help="Labels CSV")
    parser.add_argument("--embeddings", required=True, help="Embeddings NPZ (reduced or full)")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration per track (seconds)")
    parser.add_argument("--soundfont", help="Path to .sf2 soundfont for FluidSynth")
    parser.add_argument("--out-dir", required=True, help="Output directory for generated music")
    parser.add_argument("--format", choices=["midi", "wav", "both"], default="both", help="Output format")
    args = parser.parse_args()
    
    # Compute cluster features
    print("Computing cluster features...")
    feats = compute_cluster_features(Path(args.index), Path(args.clusters), Path(args.embeddings))
    
    # Save feature summary
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "cluster_music_features.json").open("w") as f:
        json.dump(feats, f, indent=2)
    print(f"Saved cluster features to {out_dir / 'cluster_music_features.json'}")
    
    # Generate music for each cluster
    for cid, feat in feats.items():
        midi_path = out_dir / f"cluster_{cid}.mid"
        wav_path = out_dir / f"cluster_{cid}.wav"
        
        if args.format in ["midi", "both"]:
            generate_midi_for_cluster(feat, args.duration, midi_path)
        
        if args.format in ["wav", "both"]:
            if not midi_path.exists():
                generate_midi_for_cluster(feat, args.duration, midi_path)
            midi_to_wav(midi_path, wav_path, args.soundfont)
    
    print(f"\nGenerated music for {len(feats)} clusters in {out_dir}")


if __name__ == "__main__":
    main()
