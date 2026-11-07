import json
import random
from pathlib import Path
import pandas as pd
from config import VIDEO_DIR, ANNOTATION_DIR, PROCESSED_DIR, SEED

def _find_annotation_file():
    candidates = [
        ANNOTATION_DIR / "train_val_videodatainfo.json",
        ANNOTATION_DIR / "videodatainfo.json",
        ANNOTATION_DIR / "msrvtt.json",
        ANNOTATION_DIR / "annotations.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    jsons = list(ANNOTATION_DIR.glob("*.json"))
    if jsons:
        return jsons[0]
    raise FileNotFoundError(
        f"No annotation JSON found in {ANNOTATION_DIR}. "
        "Place the MSR-VTT annotation JSON in data/annotations/."
    )

def load_msrvtt_annotations(json_path=None):
    path = Path(json_path) if json_path else _find_annotation_file()
    print("\nDEBUG: JSON file being loaded:", path.resolve())
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # Standard MSR-VTT annotation JSON has videos[] and sentences[].
    videos = obj.get("videos", [])
    sentences = obj.get("sentences", obj.get("captions", []))
    video_meta = {}
    for v in videos:
        vid = str(v.get("video_id", v.get("id", "")))
        if not vid:
            continue
        video_meta[vid] = {
            "category": v.get("category", v.get("domain", "unknown")),
            "video_path": str(VIDEO_DIR / f"{vid}.mp4"),
        }

    rows = []
    for s in sentences:
        vid = str(s.get("video_id", s.get("video", "")))
        caption = s.get("caption", s.get("sentence", s.get("text", "")))
        if not vid or not caption:
            continue
        meta = video_meta.get(vid, {})
        rows.append({
            "video_id": vid,
            "category": meta.get("category", "unknown"),
            "video_path": meta.get("video_path", str(VIDEO_DIR / f"{vid}.mp4")),
            "caption": str(caption).strip()
        })

    if not rows:
        raise ValueError(
            "Could not parse captions. Expected JSON with videos[] and sentences[]. "
            "Inspect your annotation JSON and adapt load_msrvtt_annotations()."
        )
    return pd.DataFrame(rows).drop_duplicates(["video_id", "caption"])

def locate_video(video_id, expected_path=None):
    if expected_path and Path(expected_path).exists():
        return str(Path(expected_path))
    extensions = [".mp4", ".avi", ".webm", ".mkv"]
    for ext in extensions:
        p = VIDEO_DIR / f"{video_id}{ext}"
        if p.exists():
            return str(p)
    # Some copies use filenames with a prefix/suffix; exact stem match is tried.
    for p in VIDEO_DIR.glob("*"):
        if p.is_file() and p.stem == str(video_id):
            return str(p)
    return None

def prepare_subset(target_videos=100, videos_per_category=5, captions_per_video=2,
                   seed=SEED, json_path=None):
    df = load_msrvtt_annotations(json_path)
    video_table = df[["video_id", "category"]].drop_duplicates()
    available_categories = sorted(video_table["category"].unique())

    if len(available_categories) >= 20:
        # Select 20 categories deterministically, then sample 5 videos/category.
        chosen_categories = available_categories[:20]
        video_table = video_table[video_table.category.isin(chosen_categories)]
        n_per_cat = videos_per_category
    else:
        # If annotation copy has fewer categories, sample up to target_videos overall.
        chosen_categories = available_categories
        n_per_cat = None

    rng = random.Random(seed)
    selected_ids = []
    if n_per_cat is not None:
        for cat in chosen_categories:
            ids = sorted(video_table[video_table.category == cat].video_id.unique().tolist())
            if len(ids) < n_per_cat:
                raise ValueError(f"Category '{cat}' has only {len(ids)} videos; need {n_per_cat}.")
            selected_ids.extend(rng.sample(ids, n_per_cat))
    else:
        ids = sorted(video_table.video_id.unique().tolist())
        if len(ids) < target_videos:
            raise ValueError(f"Only {len(ids)} unique videos found; need {target_videos}.")
        selected_ids = rng.sample(ids, target_videos)

    selected = df[df.video_id.isin(selected_ids)].copy()
    selected["video_path"] = selected.apply(
        lambda r: locate_video(r.video_id, r.video_path), axis=1
    )
    # Exactly two captions per selected video.
    selected_rows = []
    for vid in selected_ids:
        group = selected[selected.video_id == vid].drop_duplicates("caption")
        if len(group) < captions_per_video:
            raise ValueError(f"Video {vid} has only {len(group)} captions.")
        selected_rows.extend(group.head(captions_per_video).to_dict("records"))

    out = pd.DataFrame(selected_rows)
    out = out.sort_values(["category", "video_id"]).reset_index(drop=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED_DIR / "msrvtt_subset.csv", index=False)
    summary = out.groupby("category").video_id.nunique().to_dict()
    print(f"Saved {out.video_id.nunique()} videos and {len(out)} captions.")
    print(f"Categories ({len(summary)}): {summary}")
    missing = out[out.video_path.isna() | (out.video_path == "")]
    missing = missing[~missing.video_id.map(lambda x: locate_video(x))]
    if len(missing):
        print(f"WARNING: {missing.video_id.nunique()} selected videos have no local video file.")
        print("Place the matching .mp4 files in data/videos/.")
    return out

def load_subset():
    p = PROCESSED_DIR / "msrvtt_subset.csv"
    if not p.exists():
        raise FileNotFoundError("Run `python prepare_data.py` first.")
    return pd.read_csv(p)
