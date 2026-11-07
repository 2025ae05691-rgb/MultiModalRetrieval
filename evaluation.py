import numpy as np
import pandas as pd
from config import ARTIFACT_DIR, RESULTS_DIR
from data_utils import load_subset
from model_utils import load_clip, encode_texts

def rank_metrics(scores, relevant_index):
    order = np.argsort(-scores)
    rank = int(np.where(order == relevant_index)[0][0]) + 1
    return {
        "rank": rank,
        "reciprocal_rank": 1.0 / rank,
        "recall@5": float(rank <= 5),
        "recall@10": float(rank <= 10),
        "precision@5": float(1.0 / 5 if rank <= 5 else 0.0),
    }

def evaluate():
    df = load_subset()
    index = np.load(ARTIFACT_DIR / "video_index.npz", allow_pickle=True)
    ids = index["video_ids"].astype(str).tolist()
    mean_emb = index["mean_embeddings"].astype(np.float32)
    temp_emb = index["temporal_embeddings"].astype(np.float32)
    model, _, tokenizer, device = load_clip()

    # Evaluate using the second caption per video, not the caption used to train the GRU.
    eval_rows = df.groupby("video_id", sort=False).nth(1).reset_index()
    eval_rows = eval_rows[eval_rows.video_id.astype(str).isin(ids)]
    captions = eval_rows.caption.tolist()
    text_emb = encode_texts(model, tokenizer, captions, device).cpu().numpy()
    text_ids = eval_rows.video_id.astype(str).tolist()

    rows = []
    for model_name, matrix in [("Mean pooling", mean_emb), ("Temporal GRU", temp_emb)]:
        for caption, query_vec, gt_id in zip(captions, text_emb, text_ids):
            sims = matrix @ query_vec
            gt_index = ids.index(gt_id)
            metrics = rank_metrics(sims, gt_index)
            rows.append({"model": model_name, "query": caption, "ground_truth": gt_id, **metrics})
    result = pd.DataFrame(rows)
    result.to_csv(RESULTS_DIR / "query_level_metrics.csv", index=False)
    summary = result.groupby("model").agg(
        Recall_at_5=("recall@5", "mean"),
        Recall_at_10=("recall@10", "mean"),
        Precision_at_5=("precision@5", "mean"),
        MRR=("reciprocal_rank", "mean"),
        Mean_Rank=("rank", "mean"),
        Median_Rank=("rank", "median"),
    ).reset_index()
    summary.to_csv(RESULTS_DIR / "evaluation_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved results to {RESULTS_DIR}")
    return summary

if __name__ == "__main__":
    evaluate()
