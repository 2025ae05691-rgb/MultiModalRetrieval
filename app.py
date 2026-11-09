import json
from pathlib import Path
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import open_clip

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.set_page_config(page_title="MSR-VTT Video Retrieval", layout="wide")
st.title("Natural-Language Video Retrieval")
st.caption("MSR-VTT · CLIP shared embedding space")

@st.cache_resource
def load_models():
    with open(ART/"model_config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    idx = np.load(ART/"retrieval_index.npz", allow_pickle=True)
    model, _, _ = open_clip.create_model_and_transforms(
        cfg["clip_model"], pretrained=cfg["clip_pretrained"])
    tokenizer = open_clip.get_tokenizer(cfg["clip_model"])
    return idx, model.to(DEVICE).eval(), tokenizer

try:
    index, clip_model, tokenizer = load_models()
except Exception as e:
    st.error(f"Run all notebook cells through artifact creation first. Details: {e}")
    st.stop()

query = st.text_input("Natural-language query", placeholder="A person is playing guitar")
method = st.selectbox("Video representation", ["Mean Pooling", "Temporal GRU"])
top_k = st.slider("Top-K", 1, 10, 5)
if st.button("Search", type="primary") and query.strip():
    with torch.no_grad():
        tok = tokenizer([query]).to(DEVICE)
        q = F.normalize(clip_model.encode_text(tok).float(), dim=-1).cpu().numpy()[0]
    key = "mean_embeddings" if method == "Mean Pooling" else "gru_embeddings"
    gallery = index[key].astype("float32")
    gallery /= np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12)
    scores = gallery @ q
    order = np.argsort(-scores)[:top_k]
    st.subheader(f"Top {top_k} results — {method}")
    for rank, i in enumerate(order, 1):
        vid, path = str(index["video_ids"][i]), str(index["video_paths"][i])
        with st.container(border=True):
            c1,c2 = st.columns([1,2])
            with c1:
                if Path(path).exists(): st.video(path)
                else: st.warning(f"Video missing: {path}")
            with c2:
                st.markdown(f"**Rank {rank}: {vid}**")
                st.metric("Cosine similarity", f"{float(scores[i]):.4f}")
else:
    st.info("Enter a query and click Search.")
