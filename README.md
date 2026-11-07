# Assignment 2 — Natural-Language Video Retrieval Using MSR-VTT

This project implements a text-to-video retrieval system with:
1. CLIP frame embeddings with mean pooling (baseline).
2. An order-aware GRU temporal aggregator aligned to CLIP text embeddings (experimental temporal method).
3. Cosine-similarity Top-K retrieval.
4. Recall@5, Recall@10, Precision@5, MRR, Mean Rank and Median Rank.
5. Streamlit dashboard with playable local videos.

Repo: https://github.com/2025ae05691-rgb/MultiModalRetrieval

Clone repo: git clone -b feature/Vedio-Text-Retrieval https://github.com/2025ae05691-rgb/MultiModalRetrieval.git

Pull: git fetch origin git pull --ff-only origin feature/Vedio-Text-Retrieval

git commit -m "Initial commit for Vedio-Text-Retrieval"

## 1. Environment

Recommended: Python 3.10 or 3.11. A CUDA GPU is helpful but not mandatory.
Install:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip


```bash
pip install -r requirements.txt
```

## 2. Add MSR-VTT files

Place the MSR-VTT annotation JSON in:

`data/annotations/`

The code expects the standard JSON format with:
- `videos`: list containing `video_id` and optionally `category`
- `sentences`: list containing `video_id` and `caption`

Place the corresponding downloaded video files in:

`data/videos/`

Video filenames should match the annotation `video_id`, for example `video1234.mp4`.
The code also searches `.avi`, `.webm` and `.mkv`.

Important: the assignment's 100-video / 200-caption subset must use actual MSR-VTT video clips and captions. This repository does not include the copyrighted dataset files.

## 3. Prepare the 100-video subset

```bash
python prepare_data.py
```

If your JSON file is elsewhere:

```bash
python prepare_data.py --json "path/to/videodatainfo.json"
```

The script writes:

`data/processed/msrvtt_subset.csv`

It selects five videos from each of the first 20 categories in sorted category order, when at least 20 categories are present, and selects two captions per video. If the annotation copy does not include 20 categories, it selects up to 100 videos overall; document this deviation in the report.

Check that exactly 100 videos and 200 captions were selected. The script warns about missing local video files.

## 4. Build embeddings and train temporal aggregator

```bash
python build_index.py
```

For each video, the script samples eight frames and extracts frame-level CLIP embeddings. It saves:
- `artifacts/video_index.npz`
- `artifacts/temporal_gru.pt`
- `artifacts/model_config.json`

Baseline representation: L2-normalized mean of the eight frame embeddings.

Temporal representation: a unidirectional GRU processes the ordered frame-embedding sequence. Its output is projected to 512 dimensions and trained with a symmetric contrastive objective to align each video with its first caption. The second caption per video is held out for retrieval evaluation.

This is a lightweight temporal experiment, not a pretrained large video transformer. Describe it accurately as a learned temporal aggregator over pretrained CLIP frame features.

## 5. Evaluate

```bash
python evaluation.py
```

Outputs:
- `results/query_level_metrics.csv`
- `results/evaluation_summary.csv`

Metrics:
- Recall@5 / Recall@10: fraction of queries whose ground-truth video appears in the first K results.
- Precision@5: relevant results in top five divided by five. With exactly one relevant video per query, this is 0.2 if the correct video is in the top five, otherwise 0.
- MRR: mean reciprocal rank of the ground-truth video.
- Mean Rank / Median Rank: average / median position of the ground-truth video.

## 6. Run Streamlit

```bash
streamlit run app.py
```

Enter any text query, select mean pooling or temporal GRU, and view the ranked playable videos and similarity scores.

## 7. Important evaluation and reporting notes

- The code uses caption 1 to train the GRU and caption 2 as the evaluation query. This reduces direct caption leakage, but the evaluation set is small (100 queries).
- The baseline uses the pretrained CLIP shared image-text embedding space directly.
- The temporal GRU is trained on the selected subset. With only 100 videos, its result may be unstable and may not outperform mean pooling. Report actual measured results; do not fabricate improvements.
- If a video is absent or cannot be decoded, it is skipped. The resulting index may contain fewer than 100 videos; resolve missing files before submission.
- Similarity scores are cosine similarities, not calibrated probabilities.
- The two approaches should use the same video subset and same evaluation captions for a fair comparison.

## 8. Suggested report structure

1. Objective and problem definition
2. MSR-VTT dataset and sampling protocol
3. System architecture
4. Frame sampling and CLIP feature extraction
5. Mean-pooling baseline
6. Temporal GRU representation and contrastive alignment
7. Cosine-similarity retrieval
8. Evaluation metrics and results
9. Mean pooling vs temporal experiment
10. Limitations, computational cost and conclusion

Include your own screenshots from the running Virtual Lab application.
