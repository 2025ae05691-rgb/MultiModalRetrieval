import torch
import open_clip
from PIL import Image
from config import CLIP_MODEL_NAME, CLIP_PRETRAINED

def load_clip(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    model = model.to(device).eval()
    return model, preprocess, tokenizer, device

@torch.no_grad()
def encode_texts(model, tokenizer, texts, device):
    tokens = tokenizer(list(texts)).to(device)
    feats = model.encode_text(tokens)
    return torch.nn.functional.normalize(feats.float(), dim=-1)

@torch.no_grad()
def encode_images(model, preprocess, images, device):
    batch = torch.stack([preprocess(im) for im in images]).to(device)
    feats = model.encode_image(batch)
    return torch.nn.functional.normalize(feats.float(), dim=-1)
