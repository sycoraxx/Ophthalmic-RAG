"""
embed_labels.py — Pre-compute EyeCLIP Text Embeddings (v2 — Ensembled)
──────────────────────────────────────────────────────────────────────
Run this ONCE before starting the RAG pipeline.

v2 changes:
  - **Prompt ensembling**: encodes each label through 10 ophthalmic prompt
    templates and averages the embeddings for a more robust representation.
  - **Supergroup centroids**: computes centroids for hierarchical two-stage
    classification and saves them alongside the label embeddings.

Reads the cleaned disease list from `eye_diseases_list_clean.txt`, tokenizes
each label using CLIP's text encoder (with EyeCLIP fine-tuned weights if
available), and saves the resulting label list + normalized text feature
tensors + supergroup data to `data/cache/eyeclip_text_features.pt`.

At query time, `EyeClipAgent` loads this cached file instead of re-encoding
873 labels on every startup.

Usage:
    conda activate rag
    python scripts/embed_labels.py                        # uses defaults
    python scripts/embed_labels.py --weights ./checkpoints/eyeclip_visual.pt
    python scripts/embed_labels.py --labels my_labels.txt --out data/my_cache.pt
"""

import os
import sys
import argparse
import json
import torch
from src.vision import eyeclip

# Allow imports from project root when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.vision.prompt_templates import GENERIC_TEMPLATES, MODALITY_TEMPLATES
from src.vision.ophthalmic_taxonomy import build_taxonomy, get_modality_from_label

# ─── GPU Configuration ────────────────────────────────────────────────────────
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3,4")

# ─── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS   = "./models/checkpoints/eyeclip_visual_new.pt"
DEFAULT_LABELS    = "./data/processed/eye_diseases_curated.txt"
DEFAULT_OUTPUT    = "./data/cache/eyeclip_text_features.pt"
CLIP_BACKBONE     = "ViT-B/32"


def load_labels(path: str) -> list[str]:
    """Read one label per line from a text file, skipping comments."""
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                labels.append(line)
    print(f"  Loaded {len(labels)} labels from {path}")
    return labels


def encode_ensembled(
    model,
    labels: list[str],
    taxonomy: dict,
    device: str,
) -> torch.Tensor:
    """
    Encode each label using its modality-specific prompt templates.
    Averages the resulting embeddings. Returns L2-normalized text features.
    Shape: (num_labels, embed_dim)
    """
    from src.vision.prompt_templates import get_templates_for_modality
    
    num_labels = len(labels)
    all_features = []

    print(f"  Encoding {num_labels} labels with modality-aware prompt routing...")

    for i, label in enumerate(labels):
        mod = get_modality_from_label(label)
        templates = get_templates_for_modality(mod)
        
        prompts = [t.format(label) for t in templates]
        tokens = eyeclip.tokenize(prompts, truncate=True).to(device)
        
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats.mean(dim=0)
            feats = feats / feats.norm()
            all_features.append(feats)
            
        if (i+1) % 50 == 0:
            print(f"    [{i+1}/{num_labels}] Processed...")

    ensembled = torch.stack(all_features, dim=0)
    print(f"  ✓ Ensembled text features shape: {ensembled.shape}")
    return ensembled


def compute_supergroup_centroids(
    text_features: torch.Tensor,
    taxonomy: dict,
    supergroup_names: list[str],
    device: str,
) -> torch.Tensor:
    """
    Compute L2-normalized centroid embedding for each supergroup.
    Returns tensor of shape (num_supergroups, embed_dim).
    """
    centroids = []
    for sg_name in supergroup_names:
        indices = taxonomy["supergroup_to_indices"][sg_name]
        idx_tensor = torch.tensor(indices, device=device)
        sg_features = text_features[idx_tensor]
        centroid = sg_features.mean(dim=0)
        centroid = centroid / centroid.norm()
        centroids.append(centroid)
    return torch.stack(centroids)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-cache EyeCLIP text embeddings (v2 — ensembled + hierarchical)."
    )
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="Path to eyeclip_visual.pt checkpoint")
    parser.add_argument("--labels", default=DEFAULT_LABELS,
                        help="Path to cleaned disease label file (one per line)")
    parser.add_argument("--out", default=DEFAULT_OUTPUT,
                        help="Output path for cached tensor file")
    parser.add_argument("--backbone", default=CLIP_BACKBONE,
                        help="CLIP backbone architecture (default: ViT-B/32)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 1. Load CLIP backbone ────────────────────────────────────────────────
    print(f"\n[1/5] Loading EyeCLIP backbone ({args.backbone}) on {device}...")
    model, _ = eyeclip.load(args.backbone, device=device, jit=False)

    # ── 2. Optionally overlay EyeCLIP fine-tuned weights ─────────────────────
    if os.path.exists(args.weights):
        print(f"[2/5] Loading EyeCLIP fine-tuned weights from {args.weights}...")
        checkpoint = torch.load(args.weights, map_location=device)

        # EyeCLIP checkpoints use 'model_state_dict'; handle variations
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Strip common prefixes (e.g., 'module.' from DataParallel)
        cleaned = {}
        for k, v in state_dict.items():
            clean_key = k.replace("module.", "")
            cleaned[clean_key] = v

        model.load_state_dict(cleaned, strict=False)
        print(f"  Loaded EyeCLIP text encoder weights.")
    else:
        print(f"[2/5] WARNING: EyeCLIP weights not found at {args.weights}")
        print(f"       Using vanilla CLIP weights. Re-run after placing weights.")

    model.eval()

    # ── 3. Build taxonomy first to determine modality per label ──────────────
    labels = load_labels(args.labels)
    print(f"[3/5] Building ophthalmic taxonomy...")
    taxonomy = build_taxonomy(labels)
    sg_names = taxonomy["supergroup_names"]
    sg_to_indices = taxonomy["supergroup_to_indices"]

    print(f"  {len(labels)} labels → {len(sg_names)} supergroups")
    for sg_name in sg_names:
        print(f"    [{len(sg_to_indices[sg_name]):3d}] {sg_name}")

    # ── 4. Encode all labels with modality-aware prompt ensembling ───────────
    print(f"\n[4/5] Encoding labels with modality-aware prompt ensembling...")
    text_features = encode_ensembled(model, labels, taxonomy, device)

    # Compute supergroup centroids
    sg_centroids = compute_supergroup_centroids(
        text_features, taxonomy, sg_names, device
    )
    print(f"  ✓ Supergroup centroids shape: {sg_centroids.shape}")

    # ── 5. Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {
        # v1-compatible fields
        "labels": labels,
        "text_features": text_features.cpu(),
        "backbone": args.backbone,
        "weights_used": args.weights if os.path.exists(args.weights) else "vanilla_clip",
        # v2 fields — prompt ensembling
        "generic_templates": GENERIC_TEMPLATES,
        "modality_templates": MODALITY_TEMPLATES,
        # v2 fields — hierarchical classification
        "supergroup_names": sg_names,
        "supergroup_centroids": sg_centroids.cpu(),
        "supergroup_to_indices": sg_to_indices,
    }
    torch.save(payload, args.out)
    print(f"\n[5/5] ✓ Saved cached embeddings to {args.out}")
    print(f"       Contains {len(labels)} labels × {text_features.shape[1]}d features")
    print(f"       {len(sg_names)} supergroup centroids")
    print(f"       Ensembled using modality-aware prompt routing")
    print(f"\n  Next steps:")
    print(f"    1. Place EyeCLIP weights at {DEFAULT_WEIGHTS} (if not done)")
    print(f"    2. Start the RAG pipeline: streamlit run app/main.py")


if __name__ == "__main__":
    main()
