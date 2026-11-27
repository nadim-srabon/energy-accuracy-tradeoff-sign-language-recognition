"""Build the appearance cache for the 100-class ASL Citizen pilot (REBUILT cache).

This reproduces the cache FORMAT the notebooks expect (uint8 array named
`appearance`, shape (16, streams, H, W, 3), streams=1) but is NOT the original
Kaggle cache: the original shards (asl-shared-cache-building-id00/01/02) are
only retrievable with Kaggle credentials, which this PC does not have.

Documented deviations from the unknown original recipe:
  1. No signer bounding-box crop: full frames resized to 224x224 (the dataset
     class resizes to 224 at load time anyway).
  2. streams = 1 (only the appearance crop stream; the original had >=1 with
     stream 0 = signer-bbox crop, per CONFIG comments).
  3. 16 frames sampled uniformly over the decoded frame indices
     (np.linspace(0, n-1, 16), rounded, deduplicated; videos with <16 distinct
     frames repeat indices round-robin and are logged).

Resumable: existing .npz files with the right shape are skipped.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import av
import numpy as np
from PIL import Image

VIDS = r"D:\asl_citizen_download\videos_100"
OUT = r"D:\asl_cache_rebuilt"
CACHE = os.path.join(OUT, "cache")
T = 16
SIZE = 224


def sample_indices(n_frames):
    idx = np.linspace(0, n_frames - 1, T).round().astype(int)
    idx = np.unique(idx)
    if len(idx) < T:
        pad = idx[-1:] if len(idx) else np.zeros(1, int)
        while len(idx) < T:
            idx = np.concatenate([idx, pad])
        idx = np.sort(idx[:T])
    return idx


def build_one(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    dest = os.path.join(CACHE, stem + ".npz")
    if os.path.isfile(dest):
        return ("skip", stem, 0)
    try:
        with av.open(path) as container:
            stream = container.streams.video[0]
            frames = []
            for frame in container.decode(stream):
                img = frame.to_image()  # RGB PIL
                if img.size != (SIZE, SIZE):
                    img = img.resize((SIZE, SIZE), Image.BILINEAR)
                frames.append(np.asarray(img, dtype=np.uint8))
        n = len(frames)
        if n == 0:
            return ("empty", stem, 0)
        idx = sample_indices(n)
        picked = [frames[i] for i in idx]
        appearance = np.stack(picked, axis=0)[:, None, :, :, :]  # (16, 1, H, W, 3)
        assert appearance.shape == (T, 1, SIZE, SIZE, 3) and appearance.dtype == np.uint8
        np.savez_compressed(dest, appearance=appearance)
        return ("ok", stem, n)
    except Exception as e:
        return ("error", stem, repr(e))


def main():
    os.makedirs(CACHE, exist_ok=True)
    paths = sorted(
        os.path.join(VIDS, f) for f in os.listdir(VIDS) if f.lower().endswith(".mp4")
    )
    print(f"videos: {len(paths)} -> cache: {CACHE}", flush=True)
    stats = {"ok": 0, "skip": 0, "empty": 0, "error": 0}
    errors, short = [], []
    with ProcessPoolExecutor(max_workers=10) as pool:
        futs = [pool.submit(build_one, p) for p in paths]
        for i, fut in enumerate(as_completed(futs), 1):
            status, stem, info = fut.result()
            stats[status] += 1
            if status == "error":
                errors.append((stem, info))
            if status == "empty":
                errors.append((stem, "no frames decoded"))
            if i % 250 == 0:
                print(f"  {i}/{len(paths)} {stats}", flush=True)
    print("done:", stats)
    for stem, err in errors[:20]:
        print(f"  FAIL {stem}: {err}")
    manifest_path = os.path.join(OUT, "cache_manifest.json")
    existing = sorted(f for f in os.listdir(CACHE) if f.endswith(".npz"))
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cache_type": "REBUILT (not the original Kaggle shards); see script docstring",
                "layout": "one .npz per clip: appearance uint8 (16, 1, 224, 224, 3)",
                "frame_sampling": "np.linspace(0, n-1, 16) rounded deduplicated; <16-frame clips repeat last index",
                "resize": "PIL BILINEAR to 224x224, full frame (no signer bbox crop)",
                "source_videos": VIDS,
                "n_cache_files": len(existing),
                "n_source_videos": len(paths),
                "errors": errors,
            },
            f, indent=2,
        )
    print("manifest:", manifest_path)
    if stats["error"] or stats["empty"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
