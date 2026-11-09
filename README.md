# MSR-VTT Text-to-Video Retrieval

## Project Overview

This project implements a natural-language video retrieval system using
a 100-video subset of the MSR-VTT dataset. A user enters a text query,
and the system ranks candidate videos according to semantic similarity.

Two video-representation approaches are compared:

1.  **Mean Pooling** --- averages CLIP embeddings extracted from sampled
    video frames.
2.  **Temporal GRU** --- processes the ordered sequence of frame
    embeddings using a Gated Recurrent Unit (GRU).

The query is encoded by the CLIP text encoder. Text and video embeddings
are compared in a shared embedding space using cosine similarity.

## Dataset

  Item                                                   Configuration
  --------------------------------------------- ----------------------
  Categories                                                        20
  Videos per category                                                5
  Total videos                                                     100
  Caption records per video                                          2
  Total captions                                                   200
  Frames per video                                8, uniformly sampled
  CLIP model                                                  ViT-B/32  
  Embedding dimension                                              512

### Expected folder structure

``` text
Notebook-Code/
├── data/
│   ├── annotations/
│   │   └── MSRVTT_100.json
│   └── videos/
│       ├── videoXXXX.mp4
│       └── ... (100 MP4 files)
├── artifacts/
├── results/
│   └── screenshots/
└── MSR_VTT_Assignment2_End_to_End.ipynb
```

The notebook setup creates output directories. It does **not** download,
move, or extract the dataset. Place `MSRVTT_100.json` and the
corresponding MP4 files in the locations above.

## System Architecture

### Video branch

1.  Load each video.
2.  Uniformly sample 8 frames.
3.  Pass each frame through the CLIP ViT-B/32 image encoder.
4.  Extract one 512-dimensional embedding per frame.
5.  Create a video representation using Mean Pooling or Temporal GRU.

### Text branch

1.  Accept a caption or natural-language query.
2.  Tokenize it using the CLIP tokenizer.
3.  Pass it through the CLIP text encoder.
4.  Extract a 512-dimensional text embedding.

### Similarity and ranking

Text and video embeddings are L2-normalized. Their dot product is then
equivalent to cosine similarity. Candidate videos are sorted by
similarity and the highest-ranked Top-K results are returned.

For the reported evaluation, 200 caption queries are compared against
100 candidate videos, producing a 200 × 100 similarity matrix.

## Representation Methods

### 1. Mean Pooling

For frame embeddings F₁, ..., F₈:

`v_mean = Normalize((F₁ + F₂ + ... + F₈) / 8)`

This produces one normalized 512-dimensional vector per video. It is
simple and computationally inexpensive, but does not explicitly preserve
frame order.

### 2. Temporal GRU

The GRU receives an ordered frame tensor with shape:

``` text
N × 8 × 512
```

It processes the frame sequence and projects its final hidden
representation to a normalized 512-dimensional video vector.

Recorded training configuration:

  Parameter                    Value
  ------------------------- --------
  GRU hidden dimension           512
  Learning rate               0.0002
  Epochs                         100
  Weight decay                0.0001
  Contrastive temperature       0.07
  Batch size                      16
  Random seed                     42

The GRU is trained to align video representations with paired CLIP text
embeddings using symmetric contrastive loss:

`L = 0.5 × (L_video→text + L_text→video)`

## Latest Evaluation Results

  Metric           Mean Pooling   Temporal GRU
  -------------- -------------- --------------
  Recall@1                 0.60           0.10
  Precision@1              0.60           0.10
  Recall@5                 0.95           0.75
  Precision@5             0.190          0.150
  Recall@10                1.00           0.85
  Precision@10            0.100          0.085
  MRR                    0.7671         0.3668
  Mean Rank                1.80           5.55
  Median Rank               1.0            3.0

Higher Recall, Precision, and MRR are better. Lower Mean Rank and Median
Rank are better.

### Interpretation

Mean Pooling outperforms Temporal GRU on all nine reported metrics in
the latest experiment. It achieves Recall@1 of 0.60 compared with 0.10
for GRU, and Recall@10 of 1.00 compared with 0.85. Its MRR is higher
(0.7671 versus 0.3668), while its mean rank is lower (1.80 versus 5.55).

These findings are specific to the dataset and experimental
configuration. They do not establish that temporal modelling is
generally unnecessary or that GRU is inherently inferior.

## Environment and Dependencies

Below are main libraries and tools used by this type of pipeline include:

-   Python
-   PyTorch
-   OpenCLIP
-   NumPy
-   pandas
-   scikit-learn
-   The video-reading library used by the notebook (for example, OpenCV)
-   Pillow
-   Streamlit (for the retrieval interface)

## Configuration

DATA_DIR = PROJECT_ROOT / "data"
ANNOTATION_FILE = DATA_DIR / "annotations" / "MSRVTT_100.json"
VIDEO_DIR = DATA_DIR / "videos"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"
SCREENSHOT_DIR = RESULTS_DIR / "screenshots"
```

Update `PROJECT_ROOT` if the project is stored elsewhere.

## Running the Project

1.  Place `MSRVTT_100.json` in `data/annotations/`.
2.  Place the 100 corresponding MP4 files in `data/videos/`.
3.  Open `GROUP5_Multimodal_Information_Retrieval_Assignment2.ipynb` in Jupyter Notebook or JupyterLab.
4.  Confirm that `PROJECT_ROOT` points to the correct project directory.
5.  Run notebook cells in order, starting with imports and
    configuration.
6.  Complete frame extraction, CLIP feature extraction, representation
    generation, GRU training, and retrieval evaluation.
7.  Review the generated metrics and saved artifacts.
8.  If the notebook generates `app.py`, launch the Streamlit interface
    from the project directory:

``` bash
streamlit run app.py
```

9.  Enter a natural-language query, select a representation method and
    Top-K value, and inspect the retrieved videos.
10. Save application screenshots in `results/screenshots/`.

**Note:** Verify the exact generated artifact filenames and application
launch command against the final notebook. This README describes the
project workflow; it does not claim that unverified commands or files
have been executed.

## Outputs and Deliverables

The project is intended to produce:

-   CLIP frame and video features.
-   Mean-pooled video representations.
-   A trained Temporal GRU model.
-   A retrieval index or stored video embeddings.
-   Evaluation metrics comparing the two representation methods.
-   A Streamlit interface for text-to-video search.
-   Screenshots of query input and Top-K retrieval results.
-   A technical report describing the architecture, experiments,
    results, limitations, and conclusions.

Store model files and embeddings in `artifacts/`, and evaluation outputs
and screenshots in `results/`, consistent with the paths used by the
notebook.

## Limitations and Future Work

-   The experiment uses only 100 videos.
-   Eight uniformly sampled frames may miss short or important events.
-   A small training set may limit the GRU's ability to learn robust
    temporal patterns.
-   The latest results are specific to this subset and experimental
    setup.
-   Runtime and hardware costs should be reported using measurements
    from the actual execution environment.

Future work could evaluate larger video subsets, sample more frames or
temporal segments, improve temporal modelling, and use a carefully
controlled video-level train/test split to assess generalisation.

## Reproducibility Notes

-   The configured random seed is 42.
-   Record the Python, PyTorch, OpenCLIP, CUDA, and GPU versions used
    for the final run.
-   Record changes to frame sampling, model checkpoint, training
    parameters, or evaluation split.
-   Use metrics from the final executed evaluation when updating the
    report and README.
-   Do not include MSR-VTT video files in a public repository unless the
    dataset's distribution terms permit it.

## Summary

This project demonstrates a CLIP-based natural-language video retrieval
pipeline and compares mean-pooled frame representations with a GRU-based
temporal representation. In the latest reported evaluation, Mean Pooling
is the stronger method across all nine retrieval metrics. This is an
empirical finding for the current subset and configuration, not a
general conclusion that temporal modelling is unnecessary.
