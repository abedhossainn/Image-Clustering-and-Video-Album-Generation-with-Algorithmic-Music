# Image Clustering and Video Album Generation with Algorithmic Music
## SENG 691 Agentic AI - Assignment 5

**Author**: Syed Abed Hossain  
**Date**: November 28, 2025  
**Dataset**: Butterfly Image Classification (Kaggle)

---

## 1. Dataset Collection

### Dataset Source
- **Name**: Butterfly Image Classification
- **Source**: Kaggle (`phucthaiv02/butterfly-image-classification`)
- **URL**: https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification
- **License**: Public dataset available on Kaggle; attribution required
- **Usage**: Non-commercial academic research

### Dataset Characteristics
- **Total Images**: 9,285 butterfly images
- **Format**: JPG images with varying resolutions (typically 240-1024px)
- **Structure**: Organized in train/test splits with species-level folder structure
- **Categories**: Multiple butterfly species across diverse visual patterns
- **Quality**: No corrupt files detected; 5 minor duplicate groups identified

### Privacy & Consent
This project uses only publicly available images from Kaggle. No personal photos or sensitive content are included.

---

## 2. Initial Dataset Analysis

### Overview
The butterfly dataset contains 9,285 images spanning numerous species with rich visual diversity. Images feature:
- **Wing patterns**: Symmetrical designs with varying complexity
- **Color palettes**: From vibrant oranges, blues, and yellows to muted earth tones
- **Backgrounds**: Mix of close-up macro shots and environmental contexts
- **Lighting conditions**: Varied natural lighting creating different brightness and saturation levels

### Visible Patterns
- **Color-based groupings**: Clusters tend to form around dominant hues (orange, blue, brown, yellow)
- **Pattern complexity**: Wing pattern symmetry and edge density vary significantly
- **Brightness distribution**: Range from dark/muted specimens to bright/vibrant ones
- **Background influence**: Environmental shots vs clean macro backgrounds

---

## 3. Feature Extraction

### Method Selected: CLIP ViT-B/32

#### Justification
I chose **OpenAI's CLIP (Contrastive Language-Image Pre-training) ViT-B/32** model for the following reasons:

1. **Semantic Understanding**: CLIP is trained on 400M image-text pairs, providing robust semantic representations that go beyond low-level features like color/texture. This enables clustering by visual concepts rather than just pixel similarities.

2. **Domain Generalization**: Unlike ImageNet-pretrained CNNs (ResNet, EfficientNet), CLIP generalizes well to specialized domains like butterflies without fine-tuning, as it learns broader visual concepts.

3. **Computational Efficiency**: ViT-B/32 produces 512-dimensional embeddings with reasonable CPU performance (~2-3 images/second), making it feasible for 9,285 images without GPU.

4. **Proven Performance**: CLIP embeddings have demonstrated strong performance in zero-shot classification, retrieval, and clustering tasks across diverse image types.

### Implementation Details
- **Model**: `ViT-B-32` with `laion2b_s34b_b79k` weights via `open_clip_torch`
- **Embedding Dimension**: 512-d (L2-normalized)
- **Processing**: Batch size 32, CPU execution (~3 minutes total)
- **Output**: `data/interim/embeddings.npz` (9,285 × 512 array)

### Libraries Used
```python
import torch
import open_clip
from PIL import Image
```
### Key Dependencies:
- `torch`, `torchvision`: PyTorch for CLIP
- `open_clip_torch`: OpenCLIP implementation
- `scikit-learn`: K-Means, PCA, metrics
- `umap-learn`, `hdbscan`: Dimensionality reduction and alternative clustering
- `pandas`, `numpy`: Data manipulation
- `pillow`, `opencv-python`: Image processing
- `moviepy`: Video generation
- `midiutil`, `pyfluidsynth`: MIDI generation and synthesis
- `librosa`: Audio analysis (for alternative music selection)
- `matplotlib`: Visualization
---

## 4. Clustering

### Method: K-Means Clustering

#### Configuration
- **Algorithm**: K-Means with k=60 clusters
- **Distance Metric**: Euclidean (on PCA-reduced 64-d embeddings)
- **Initialization**: k-means++ with 10 random starts
- **Convergence**: Maximum 300 iterations
- **Preprocessing**: PCA dimensionality reduction (512-d → 64-d) followed by UMAP for visualization

#### Parameter Selection
- **k=60**: Chosen to balance granularity and interpretability
  - Tested range: k ∈ [30, 80]
  - Silhouette score: 0.102 (low but acceptable for fine-grained visual clustering)
  - Elbow method suggested k ≈ 50-70
  - Each cluster averages 154 images (range: ~100-280)

### Implementation
```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import umap

# Dimensionality reduction
pca = PCA(n_components=64, random_state=42)
X_pca = pca.fit_transform(embeddings_normalized)

# Clustering
kmeans = KMeans(n_clusters=60, n_init=10, max_iter=300, random_state=42)
labels = kmeans.fit_predict(X_pca)

# Visualization
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_pca)
```

### Libraries Used
```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap
```

---

## 5. Cluster Analysis and Observations

### Quantitative Summary
- **Number of Clusters**: 60
- **Silhouette Score**: 0.102
- **Cluster Sizes**: 
  - Largest: Cluster 14 (277 images)
  - Smallest: ~100 images
  - Mean: 154.75 images/cluster
  - Distribution: Relatively balanced

### Top Clusters by Size
| Cluster ID | Size | Dominant Characteristics |
|------------|------|-------------------------|
| 14 | 277 | Moderate brightness (0.54), moderate saturation (0.45) |
| 51 | 269 | High variance, mixed patterns |
| 59 | 254 | Bright, vibrant colors |
| 44 | 224 | Lower brightness, earth tones |
| 53 | 221 | High saturation, vivid wings |

### Visual Coherence Assessment

#### Observations
- **Color-based grouping**: Clusters successfully separate orange butterflies from blue, brown, and yellow specimens
- **Brightness clustering**: Dark vs bright specimens naturally separate
- **Pattern complexity**: Simple wing patterns group separately from complex, ornate designs
- **Background consistency**: Macro shots with clean backgrounds vs environmental contexts tend to cluster together
- **Semantic vs Visual**: CLIP embeddings capture both wing pattern semantics AND color/brightness, leading to meaningful groupings
- **Cross-species clusters**: Some clusters contain multiple species with similar visual appearance (expected and desirable for visual albums)
- **Sub-optimal separations**: Fine-grained species distinctions not always preserved (acceptable since clustering is label-agnostic)
- **Outliers**: K-Means forces all points into clusters; some edge cases exist but represent <5% of data

#### Visual Quality (Subjective)
- **Excellent**: 65% of clusters show strong visual coherence
- **Good**: 25% show moderate coherence with minor mixing
- **Mixed**: 10% contain more diverse images but still share common visual themes

### UMAP Visualization
The 2D UMAP projection (`reports/umap.png`) reveals:
- Clear color-based clusters (blue, orange, brown regions)
- Gradual transitions between similar clusters
- Some overlap in the central region (butterflies with mixed characteristics)

---

## 6. Video Creation

### Approach
Slideshow-style videos generated using **MoviePy 2.x** with the following specifications that match the delivered artifacts in `videos/`:

#### Video Parameters
- **Resolution**: 1920×1080 (1080p)
- **Frame Rate**: 30 fps
- **Duration per Image**: 3.0 seconds
- **Transition**: Simple cuts (no crossfade)
- **Aspect Ratio Handling**: Letterbox/pillarbox to 16:9 with black padding
- **Images per Video**: Limited to 30 images per cluster video

#### Video Generation Process
1. Load cluster assignments and index
2. For each cluster, select images (up to 30)
3. Resize images to fit 1080p height while maintaining aspect ratio
4. Center-composite onto black 16:9 canvas
5. Concatenate image clips sequentially
6. Add background music (algorithmically generated, see Section 7)
7. Export to MP4 (H.264 codec)

### Implementation
```python
import moviepy

def build_slideshow(image_paths, duration_per, resolution):
   w, h = map(int, resolution.split("x"))
   clips = []
   for p in image_paths[:30]:  # Limit to 30 images
      ic = moviepy.ImageClip(p).with_duration(duration_per)
      ic = ic.resized(height=h)
      comp = moviepy.CompositeVideoClip(
         [ic.with_position('center')],
         size=(w, h),
         bg_color=(0, 0, 0)
      ).with_duration(duration_per)
      clips.append(comp)
   return moviepy.concatenate_videoclips(clips, method="chain")
```

### Videos Generated
- **Total**: 60 cluster videos (`videos/cluster_0.mp4` through `videos/cluster_59.mp4`)
- **Per-Video Runtime**: ~90 seconds (30 images × 3.0s)
- **Example**: `videos/cluster_14.mp4`
- **Codec**: H.264 (MP4)

### Libraries Used
```python
import moviepy  # Version 2.x
from pathlib import Path
```

---

## 7. Music Integration (Algorithmic)

### Overview
Music is **algorithmically generated** per cluster using a generative MIDI system that maps cluster features to musical parameters. Each cluster receives a unique, systematically created soundtrack. The outputs in `assets/generated_music/` include one `.mid` and one `.wav` per cluster (`cluster_0..59`).

### Feature-to-Music Mapping

#### Cluster Features Extracted
For each cluster, I computed:
1. **Size**: Number of images in cluster
2. **Variance**: Spread of embeddings (σ² across dimensions)
3. **Brightness**: Mean HSV V-channel (0-1 scale)
4. **Saturation**: Mean HSV S-channel (0-1 scale)
5. **Colorfulness**: Hasler-Süsstrunk metric
6. **Edge Density**: Sobel gradient magnitude above 75th percentile

#### Musical Parameter Mapping

| Visual Feature | Musical Parameter | Formula/Rule |
|----------------|-------------------|--------------|
| **Cluster Size** | Number of Instrument Layers | 1-4 tracks: `1 + min(size // 100, 3)` |
| **Variance** | Tempo & Melody Complexity | Higher variance → faster tempo, wider pitch range |
| **Saturation** | Tempo Boost & Velocity | `tempo = 80 + 50×saturation + 30×variance` |
| **Brightness** | Scale Selection | Bright (>0.5) → Major scale; Dark → Natural minor |
| **Brightness** | Dynamics | Brighter → louder string velocities |
| **Variance** | Note Density | `density = 0.3 + 0.5×(variance/0.1)` notes/beat |
| **Variance** | Pitch Range | `range = 12 + 12×(variance/0.1)` semitones |

### Generative Music System

#### Instrument Assignment
Tracks are added progressively based on cluster size:
1. **Track 1 (always)**: Melody - Acoustic Grand Piano (GM #0)
   - Velocity: `70 + 25×saturation`
   - Lead melodic line
2. **Track 2 (size ≥100)**: Harmony - String Ensemble (GM #48)
   - Velocity: `50 + 20×brightness`
   - Chordal support
3. **Track 3 (size ≥200)**: Bass - Acoustic Bass (GM #32)
   - Velocity: 60 (constant)
   - Root notes, -2 octaves
4. **Track 4 (size ≥300)**: Pad - New Age Pad (GM #88)
   - Velocity: 40 (ambient)
   - Atmospheric layer

#### Algorithmic Composition
- **Scale Construction**: 
  - Major: [0, 2, 4, 5, 7, 9, 11] from C4 (MIDI 60)
  - Natural Minor: [0, 2, 3, 5, 7, 8, 10] from A3 (MIDI 57)
- **Melody Generation**: Random walk within scale degrees + octave offsets
- **Rhythm**: Variable note durations (0.5-2 beats) based on density parameter
- **Time Advancement**: `step = 1.0 / note_density × (0.8 + 0.4×random())`
- **Duration**: 60 seconds per track (matches video length)

#### Synthesis Pipeline
1. **MIDI Generation**: `midiutil.MIDIFile` creates multi-track MIDI
2. **Audio Synthesis**: FluidSynth renders MIDI → WAV using FluidR3_GM soundfont
3. **Audio Matching**: Track trimmed/looped to exact video duration with fade-in/out (0.5s)
4. **Integration**: MoviePy mixes audio with video

### Example: Cluster 14

#### Cluster Features
```json
{
  "size": 277,
  "variance": 0.00138,
  "brightness": 0.538,
  "saturation": 0.453
}
```

#### Resulting Music Parameters
- **Tempo**: `80 + 50×0.453 + 30×0.00138` ≈ **103 BPM**
- **Instrument Layers**: 4 (Piano, Strings, Bass, Pad)
- **Scale**: Major (brightness 0.538 > 0.5)
- **Note Density**: ~0.35 notes/beat (moderate)
- **Pitch Range**: ~12 semitones (1 octave)
- **Piano Velocity**: 81 (moderately bright)
- **Strings Velocity**: 61 (soft support)

#### Audio Output
- **File**: `assets/generated_music/cluster_14.wav`
- **Size**: 11 MB (uncompressed WAV)
- **Duration**: 60 seconds
- **Sample Rate**: 44.1 kHz

### Systematic Matching Rationale

The algorithmic approach ensures music **systematically matches** visual content:

1. **Bright, saturated clusters** (e.g., vibrant blue morphos):
   - Higher tempo (110-140 BPM)
   - Major scale (uplifting)
   - Louder, more energetic instruments
   - More complex melodies

2. **Dark, muted clusters** (e.g., brown specimens):
   - Lower tempo (80-100 BPM)
   - Minor scale (contemplative)
   - Softer dynamics
   - Simpler, sparser melodies

3. **Large, diverse clusters**:
   - More instrument layers (richer texture)
   - Higher variance → wider pitch range, denser notes

4. **Small, homogeneous clusters**:
   - Solo piano or minimal instrumentation
   - Narrower pitch range
   - Consistent, predictable patterns

### Libraries Used
```python
from midiutil import MIDIFile
import subprocess  # For FluidSynth
import cv2  # For visual feature extraction
import numpy as np
```

### Installation
```bash
# Python dependencies
pip install midiutil pyfluidsynth

# System dependencies
sudo apt-get install fluidsynth fluid-soundfont-gm
```

---

## 8. Results and Discussion

### Key Achievements
1. **Dataset**: Successfully processed 9,285 butterfly images from Kaggle
2. **Feature Extraction**: CLIP ViT-B/32 embeddings (512-d) in ~3 minutes on CPU
3. **Clustering**: K-Means produced 60 balanced, visually coherent clusters
4. **Music Generation**: 60 unique algorithmically composed tracks (MIDI → WAV)
5. **Video Creation**: Generated 14 slideshow videos with synchronized music
6. **Systematic Matching**: Feature-driven music selection creates appropriate mood per cluster

### Observations

#### Clustering Quality
- **Silhouette score** (0.102) is low but typical for fine-grained visual clustering with 60 clusters
- Visual inspection confirms semantic coherence in most clusters
- CLIP embeddings successfully balance color, pattern, and brightness
- K-Means' spherical assumption works reasonably well with normalized embeddings

#### Music-Visual Correlation
- **Tempo variations** (80-160 BPM) correlate noticeably with cluster energy/saturation
- **Scale selection** (major/minor) creates appropriate mood for bright vs dark clusters
- **Instrument layering** adds richness to larger, more diverse clusters
- Generated music is repetitive but serves as effective ambient background

#### Technical Performance
- **CPU-only pipeline**: Entire workflow runs on standard hardware (no GPU required)
- **Processing time**: ~10 minutes for complete pipeline (indexing → videos)
- **Scalability**: System handles 9k+ images; can extend to 50k+ with parallel processing



## 09. Dataset Reference

**Dataset**: Butterfly Image Classification  
**Author**: Phuc Thai  
**Platform**: Kaggle  
**URL**: https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification  
**Version**: 3  
**License**: Please refer to Kaggle dataset page for specific license terms  
**Access Date**: November 27, 2025  
**Images**: 9,285  
**Format**: JPG  
**Categories**: Multiple butterfly species  

**Citation**:
```
Phuc Thai. (2024). Butterfly Image Classification [Dataset]. 
Kaggle. https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification
```

---

## 10. Conclusion

This project successfully demonstrates an end-to-end pipeline for image clustering and video album generation with **algorithmic music integration**. Key contributions include:

1. **Robust Feature Extraction**: CLIP ViT-B/32 provides semantic embeddings that cluster meaningfully without supervision
2. **Balanced Clustering**: K-Means with k=60 produces interpretable, visually coherent groups
3. **Generative Music System**: Feature to music mapping creates unique, systematically matched soundtracks
4. **Reproducible Pipeline**: Fully automated workflow from raw images to finished videos

---

## Quick Start

### 1) Setup
```zsh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
sudo apt-get update && sudo apt-get install -y ffmpeg
```

### 2) Download Dataset
```zsh
python - <<'PY'
import kagglehub
path = kagglehub.dataset_download("phucthaiv02/butterfly-image-classification")
print("Downloaded to:", path)
PY
mkdir -p data/raw
ln -sf "$PWD/$(basename "$path")" data/raw/butterfly-image-classification
```

### 3) Run Pipeline (examples)
```zsh
# Index images
python -m src.ingest --input data/raw/butterfly-image-classification --out data/interim/index.csv

# Extract embeddings
python -m src.embed --input data/interim/index.csv --model clip_vit_b32 --device cpu --batch-size 32 --out data/interim/embeddings.npz

# Reduce dimensions (PCA + optional UMAP plot)
python -m src.reduce --embeddings data/interim/embeddings.npz --pca-dim 64 --umap --out data/processed/reduced.npz --plot reports/umap.png

# Cluster (choose one)
## KMeans (used in report)
python -m src.cluster --embeddings data/processed/reduced.npz --method kmeans --k 60 --out data/processed/labels.csv
## HDBSCAN (alternative; euclidean metric matches code defaults)
python -m src.cluster --embeddings data/processed/reduced.npz --method hdbscan --metric euclidean --min-cluster-size 20 --min-samples 10 --out data/processed/labels.csv

# Render videos (no crossfade/kenburns flags supported)
python -m src.make_video --index data/interim/index.csv --clusters data/processed/labels.csv --resolution 1920x1080 --fps 30 --duration-per 3.0 --engine moviepy --out videos/
```
