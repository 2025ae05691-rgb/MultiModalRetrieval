# Assignment 2: Natural-Language Video Retrieval Using MSR-VTT

## 1. Objective
Describe the goal of retrieving relevant video clips from natural-language queries.

## 2. Dataset
- Dataset: MSR-VTT
- Categories:
- Number of videos actually indexed:
- Number of captions:
- Sampling seed:
- Missing/unreadable videos:

## 3. Architecture
Insert your system architecture diagram. Explain the video and text pipelines, shared CLIP embedding space, similarity search and Streamlit interface.

## 4. Video preprocessing and feature extraction
Describe frame sampling, number of frames, CLIP model and preprocessing.

## 5. Retrieval methods
### 5.1 Baseline: mean pooling
Explain averaging and L2 normalization of frame embeddings.

### 5.2 Temporal method: GRU
Explain ordered frame input, GRU aggregation, projection and contrastive alignment with caption 1.

## 6. Evaluation protocol
Explain why caption 1 is used for training and caption 2 for evaluation. Define Recall@K, Precision@K, MRR, Mean Rank and Median Rank.

## 7. Results
Paste `results/evaluation_summary.csv` and create a comparison chart.

## 8. Research analysis
Answer using your actual results:
- Does temporal modeling improve retrieval?
- Which metrics improved or declined?
- What is the computational trade-off?
- Which categories benefit?
- Is mean pooling adequate for this 100-video dataset?

## 9. Limitations
Discuss small subset size, video availability, model size, training stability and generalization.

## 10. Conclusion
Summarize findings supported by your measured results.

## 11. Evidence
Insert screenshots of query execution, Top-K playable results, model selection and evaluation charts.
