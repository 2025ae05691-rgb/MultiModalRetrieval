import numpy as np
import pandas as pd
import streamlit as st
import torch
from pathlib import Path
from config import ARTIFACT_DIR, RESULTS_DIR
from model_utils import load_clip, encode_texts

st.set_page_config(page_title="MSR-VTT Video Search", page_icon="🎬", layout="wide")
st.title("🎬 Natural-Language Video Retrieval")
st.caption("MSR-VTT | Search video clips using natural-language descriptions")

@st.cache_resource
def resources():
    model, _, tokenizer, device = load_clip()
    return model, tokenizer, device

@st.cache_data
def load_index():
    p = ARTIFACT_DIR / "video_index.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return {
        "video_ids": z["video_ids"].astype(str),
        "categories": z["categories"].astype(str),
        "paths": z["paths"].astype(str),
        "mean": z["mean_embeddings"].astype(np.float32),
        "temporal": z["temporal_embeddings"].astype(np.float32)
    }

idx = load_index()
if idx is None:
    st.error("Video index not found. Run `python build_index.py` before starting Streamlit.")
    st.stop()

with st.sidebar:
    st.header("Search settings")
    method = st.selectbox("Representation", ["Mean pooling (CLIP baseline)", "Temporal GRU"])
    top_k = st.slider("Number of results (K)", 1, min(20, len(idx["video_ids"])), 5)
    st.info(f"Indexed videos: {len(idx['video_ids'])}")

query = st.text_input("Describe the video you want to find",
                      placeholder="A person is playing guitar on a stage.")
search = st.button("Search videos", type="primary", disabled=not query.strip())

if search:
    with st.spinner("Encoding query and searching videos..."):
        model, tokenizer, device = resources()
        q = encode_texts(model, tokenizer, [query], device).cpu().numpy()[0]
        matrix = idx["mean"] if method.startswith("Mean") else idx["temporal"]
        scores = matrix @ q
        order = np.argsort(-scores)[:top_k]

    st.subheader(f"Top {top_k} matching videos")
    cols = st.columns(2)
    for rank, i in enumerate(order, start=1):
        path = Path(idx["paths"][i])
        with cols[(rank - 1) % 2]:
            st.markdown(f"**#{rank} · {idx['video_ids'][i]}**")
            st.caption(f"Category: {idx['categories'][i]} | Cosine similarity: {scores[i]:.4f}")
            if path.exists():
                st.video(str(path))
            else:
                st.warning(f"Video file is missing: {path}")
            st.progress(float(np.clip((scores[i] + 1) / 2, 0, 1)),
                        text="Similarity (scaled for display)")

st.divider()
st.subheader("Evaluation results")
summary_path = RESULTS_DIR / "evaluation_summary.csv"
if summary_path.exists():
    summary = pd.read_csv(summary_path)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.bar_chart(summary.set_index("model")[["Recall_at_5", "Recall_at_10", "MRR"]])
else:
    st.info("Run `python evaluation.py` to generate the baseline vs temporal evaluation table.")
