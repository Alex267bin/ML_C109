"""K-Means image segmentation in RGB color space."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage import data


def load_image(image_path: Path | None, max_size: int = 256) -> Image.Image:
    if image_path and image_path.exists():
        image = Image.open(image_path).convert("RGB")
    else:
        # Reproducible natural portrait supplied by scikit-image for a no-argument demo.
        image = Image.fromarray(data.astronaut()).convert("RGB")
    image.thumbnail((max_size, max_size))
    return image


def segment_image(image_path: Path | None, output_dir: Path, k: int = 5, max_size: int = 256) -> Path:
    if k < 2:
        raise ValueError("K must be at least 2.")
    output_dir.mkdir(parents=True, exist_ok=True)
    original = load_image(image_path, max_size=max_size)
    pixels = np.asarray(original, dtype=np.float32)
    height, width, channels = pixels.shape
    X = pixels.reshape(-1, channels)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    segmented = model.cluster_centers_[labels].reshape(height, width, channels)
    segmented = np.clip(segmented, 0, 255).astype(np.uint8)

    segmented_path = output_dir / f"segmented_k{k}.png"
    comparison_path = output_dir / f"comparison_k{k}.png"
    Image.fromarray(segmented).save(segmented_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(np.asarray(original))
    axes[0].set_title("Original image")
    axes[1].imshow(segmented)
    axes[1].set_title(f"K-Means segmentation (K={k})")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(comparison_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return comparison_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment an image using K-Means.")
    parser.add_argument("--image", type=Path, default=None, help="Optional input image path.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("outputs/segmented_images"))
    args = parser.parse_args()
    print(f"Saved comparison: {segment_image(args.image, args.output, args.k, args.max_size)}")


if __name__ == "__main__":
    main()
