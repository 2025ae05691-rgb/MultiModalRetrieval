
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from config import (
    ARTIFACT_DIR,
    NUM_FRAMES,
    EMBEDDING_DIM,
    TEMPORAL_HIDDEN_DIM,
    TEMPORAL_LEARNING_RATE,
    TEMPORAL_EPOCHS,
    TEMPORAL_WEIGHT_DECAY,
    TEMPERATURE,
    SEED
)

from data_utils import load_subset, locate_video
from model_utils import load_clip, encode_texts
from video_utils import sample_video_frames


# ==========================================
# REPRODUCIBILITY
# ==========================================

torch.manual_seed(SEED)
np.random.seed(SEED)


# ==========================================
# TEMPORAL GRU MODEL
# ==========================================

class TemporalGRU(nn.Module):

    def __init__(
        self,
        dim=EMBEDDING_DIM,
        hidden=TEMPORAL_HIDDEN_DIM
    ):

        super().__init__()

        # Processes frame embeddings sequentially
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=False
        )

        # Projects GRU output into CLIP embedding space
        self.proj = nn.Linear(
            hidden,
            dim
        )

    def forward(self, x):

        # x shape:
        # (batch_size, 8 frames, 512 dimensions)

        _, hidden_state = self.gru(x)

        # Last GRU hidden state
        final_state = hidden_state[-1]

        # Project into 512-dimensional space
        output = self.proj(final_state)

        # L2 normalization
        output = F.normalize(
            output,
            dim=-1
        )

        return output


# ==========================================
# MAIN FEATURE EXTRACTION PIPELINE
# ==========================================

def main():

    print("\nStarting Video Feature Extraction\n")

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------
    # STEP 1: Load prepared dataset
    # --------------------------------------

    df = load_subset()

    videos = (
        df[
            ["video_id", "category", "video_path"]
        ]
        .drop_duplicates(subset=["video_id"])
        .reset_index(drop=True)
    )

    print("Total videos in dataset:", len(videos))

    # --------------------------------------
    # STEP 2: Load pretrained CLIP
    # --------------------------------------

    model, preprocess, tokenizer, device = load_clip()

    print("CLIP loaded successfully")
    print("Device:", device)

    # --------------------------------------
    # STEP 3: Initialize storage
    # --------------------------------------

    # Stores original frame-level embeddings
    frame_sequences = []

    # Stores mean-pooled video embeddings
    mean_vectors = []

    # Stores video metadata
    valid_videos = []

    # --------------------------------------
    # STEP 4: Extract frames and embeddings
    # --------------------------------------

    for row in tqdm(
        videos.itertuples(index=False),
        total=len(videos),
        desc="Processing videos"
    ):

        video_id = str(row.video_id)

        video_path = locate_video(
            video_id,
            row.video_path
        )

        if not video_path:

            print(
                f"\nSkipping missing video: {video_id}"
            )

            continue

        try:

            # Extract 8 uniformly sampled frames
            frames = sample_video_frames(
                video_path,
                NUM_FRAMES
            )

            # Preprocess all frames
            pixels = torch.stack([
                preprocess(frame)
                for frame in frames
            ]).to(device)

            # ----------------------------------
            # CLIP IMAGE FEATURE EXTRACTION
            # ----------------------------------

            with torch.no_grad():

                feats = model.encode_image(
                    pixels
                ).float()

                # Normalize each frame embedding
                feats = F.normalize(
                    feats,
                    dim=-1
                )

            # Expected shape: (8, 512)
            assert feats.shape == (
                NUM_FRAMES,
                EMBEDDING_DIM
            ), f"Unexpected frame feature shape: {feats.shape}"

            # ----------------------------------
            # BRANCH A: MEAN POOLING
            # ----------------------------------

            mean_embedding = feats.mean(
                dim=0
            )

            mean_embedding = F.normalize(
                mean_embedding,
                dim=0
            )

            # Store one video embedding
            mean_vectors.append(
                mean_embedding.cpu().numpy()
            )

            # ----------------------------------
            # BRANCH B: TEMPORAL FEATURES
            # ----------------------------------

            # IMPORTANT:
            # Preserve all 8 individual frame embeddings.
            # Do NOT use mean_embedding here.

            frame_sequences.append(
                feats.cpu()
            )

            # Store video metadata
            valid_videos.append({

                "video_id": video_id,

                "category": row.category,

                "video_path": video_path

            })

        except Exception as e:

            print(
                f"\nError processing {video_id}: {e}"
            )

            continue

    # --------------------------------------
    # STEP 5: Validate extracted features
    # --------------------------------------

    if len(valid_videos) < 2:

        raise RuntimeError(
            "Not enough videos processed. "
            "Check video files and preprocessing."
        )

    print("\nFeature extraction completed")

    print(
        "Successfully processed videos:",
        len(valid_videos)
    )

    # Shape: (N, 8, 512)
    seq = torch.stack(
        frame_sequences
    )

    # Shape: (N, 512)
    mean_embeddings = np.asarray(
        mean_vectors,
        dtype=np.float32
    )

    print("\nEmbedding dimensions:")

    print(
        "Frame-level embeddings:",
        tuple(seq.shape)
    )

    print(
        "Mean-pooled embeddings:",
        mean_embeddings.shape
    )

    # --------------------------------------
    # STEP 6: Save frame-level embeddings
    # --------------------------------------

    # Save all 8 frame embeddings for every video.
    # This allows future experiments without
    # extracting video frames again.

    frame_embedding_path = (
        ARTIFACT_DIR / "frame_embeddings.npy"
    )

    np.save(
        frame_embedding_path,
        seq.numpy()
    )

    print(
        "\nSaved frame embeddings:",
        frame_embedding_path
    )

    # --------------------------------------
    # STEP 7: Prepare text supervision
    # --------------------------------------

    # Use first caption of each video for GRU training.
    # Second caption is reserved for evaluation.

    caption_by_video = (
        df.groupby("video_id").caption.apply(list).to_dict()
    )

    train_captions = [
        caption_by_video[v["video_id"]][0]
        for v in valid_videos
    ]

    text_train = encode_texts(
        model,
        tokenizer,
        train_captions,
        device
    ).cpu()

    print(
        "Text training embeddings:",
        tuple(text_train.shape)
    )

    # --------------------------------------
    # STEP 8: Initialize temporal GRU
    # --------------------------------------

    temporal = TemporalGRU(
        dim=EMBEDDING_DIM,
        hidden=TEMPORAL_HIDDEN_DIM
    ).to(device)

    optimizer = torch.optim.AdamW(
        temporal.parameters(),

        lr=TEMPORAL_LEARNING_RATE,

        weight_decay=TEMPORAL_WEIGHT_DECAY
    )

    # Move data to selected device
    seq_device = seq.to(device)

    text_device = text_train.to(device)

    # --------------------------------------
    # STEP 9: Train temporal model
    # --------------------------------------

    print("\nStarting Temporal GRU Training\n")

    temporal.train()

    labels = torch.arange(
        len(valid_videos),
        device=device
    )

    for epoch in range(TEMPORAL_EPOCHS):

        optimizer.zero_grad()

        # Input: (N, 8, 512)
        # Output: (N, 512)

        video_vectors = temporal(
            seq_device
        )

        # Similarity between video and text vectors
        logits = (
            video_vectors @ text_device.T
        ) / TEMPERATURE

        # Symmetric contrastive loss
        loss_video_to_text = F.cross_entropy(
            logits,
            labels
        )

        loss_text_to_video = F.cross_entropy(
            logits.T,
            labels
        )

        loss = (
            loss_video_to_text +
            loss_text_to_video
        ) / 2

        loss.backward()

        optimizer.step()

        if (epoch + 1) % 20 == 0:

            print(
                f"Epoch {epoch + 1}/{TEMPORAL_EPOCHS} "
                f"| Loss: {loss.item():.4f}"
            )

    # --------------------------------------
    # STEP 10: Generate temporal embeddings
    # --------------------------------------

    temporal.eval()

    with torch.no_grad():

        temporal_embeddings = temporal(
            seq_device
        ).cpu().numpy()

    print(
        "\nTemporal embeddings shape:",
        temporal_embeddings.shape
    )

    # --------------------------------------
    # STEP 11: Prepare metadata
    # --------------------------------------

    video_ids = [
        v["video_id"]
        for v in valid_videos
    ]

    categories = [
        v["category"]
        for v in valid_videos
    ]

    paths = [
        v["video_path"]
        for v in valid_videos
    ]

    # --------------------------------------
    # STEP 12: Save complete video index
    # --------------------------------------

    index_path = (
        ARTIFACT_DIR / "video_index.npz"
    )

    np.savez_compressed(

        index_path,

        video_ids=np.array(video_ids),

        categories=np.array(categories),

        paths=np.array(paths),

        mean_embeddings=mean_embeddings,

        temporal_embeddings=temporal_embeddings

    )

    print(
        "\nSaved video index:",
        index_path
    )

    # --------------------------------------
    # STEP 13: Save trained GRU model
    # --------------------------------------

    torch.save(

        temporal.state_dict(),

        ARTIFACT_DIR / "temporal_gru.pt"

    )

    print(
        "Saved temporal GRU weights"
    )

    # --------------------------------------
    # STEP 14: Save model configuration
    # --------------------------------------

    model_config = {

        "clip_model": "ViT-B-32",

        "clip_pretrained": "laion2b_s34b_b79k",

        "embedding_dimension": EMBEDDING_DIM,

        "num_frames": NUM_FRAMES,

        "temporal_model": "GRU",

        "temporal_hidden_dimension": TEMPORAL_HIDDEN_DIM,

        "temporal_epochs": TEMPORAL_EPOCHS,

        "videos_indexed": len(valid_videos)

    }

    with open(

        ARTIFACT_DIR / "model_config.json",

        "w"

    ) as f:

        json.dump(
            model_config,
            f,
            indent=4
        )

    # --------------------------------------
    # FINAL SUMMARY
    # --------------------------------------

    print("\n===================================")

    print("VIDEO INDEXING COMPLETED")

    print("===================================")

    print(
        "Videos indexed:",
        len(valid_videos)
    )

    print(
        "Frame embeddings:",
        tuple(seq.shape)
    )

    print(
        "Mean embeddings:",
        mean_embeddings.shape
    )

    print(
        "Temporal embeddings:",
        temporal_embeddings.shape
    )

    print("\nGenerated files:")

    print("1. frame_embeddings.npy")

    print("2. video_index.npz")

    print("3. temporal_gru.pt")

    print("4. model_config.json")


if __name__ == "__main__":

    main()
