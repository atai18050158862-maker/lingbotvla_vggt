"""Frozen VGGT teacher for multi-view, current/future relational distillation.

Only the VGGT Aggregator (alternating frame/global attention backbone) is used;
camera/depth/point/track heads are not needed because the student is supervised
with pairwise relations of the aggregator's patch features.

Frame order fed to VGGT is [view_0 @ t, ..., view_{V-1} @ t, view_0 @ t+T, ...],
so the first configured view at the current step (the static top camera by
default) is VGGT's reference frame.
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_VGGT_PACKAGE_DIR = Path(__file__).resolve().parent / "vggt"
if _VGGT_PACKAGE_DIR.exists() and str(_VGGT_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_VGGT_PACKAGE_DIR))

VGGT_PATCH_SIZE = 14
VGGT_DEPTH = 24
VGGT_EMBED_DIM = 1024


def vggt_views(vggt_cfg):
    return [int(v) for v in vggt_cfg.get("views", [0, 1, 2])]


def _load_vggt_state_dict(ckpt_path):
    path = Path(ckpt_path)
    if path.is_dir():
        for name in ("model.safetensors", "model.pt", "model.pth"):
            if (path / name).exists():
                path = path / name
                break
        else:
            raise FileNotFoundError(f"No model.safetensors / model.pt found under {ckpt_path}.")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(path), device="cpu")
    else:
        state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
        if "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]

    prefix = "aggregator."
    if any(k.startswith(prefix) for k in state_dict):
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    return state_dict


def build_vggt_model(vggt_cfg):
    from vggt.models.aggregator import Aggregator

    ckpt_path = vggt_cfg.get("ckpt_path")
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"align_params.vggt.ckpt_path not found: {ckpt_path!r}")

    layer = int(vggt_cfg.get("layer", -1))
    if layer < 0:
        layer += VGGT_DEPTH
    if not 0 <= layer < VGGT_DEPTH:
        raise ValueError(f"align_params.vggt.layer must be in [-{VGGT_DEPTH}, {VGGT_DEPTH}), got {vggt_cfg.get('layer')}.")

    model = Aggregator(
        img_size=518,
        patch_size=VGGT_PATCH_SIZE,
        embed_dim=VGGT_EMBED_DIM,
        cached_layer_indices=(layer,),
    )
    missing, unexpected = model.load_state_dict(_load_vggt_state_dict(ckpt_path), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"VGGT aggregator checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    for p in model.parameters():
        p.requires_grad = False
    model.vggt_layer = layer
    model.cuda()
    model.eval()
    return model


def _resize_for_vggt(frames, input_size, keep_aspect):
    """frames: (N, 3, H, W) float in [0, 1] -> (N, 3, H', W') with H', W' multiples of 14."""
    _, _, h, w = frames.shape
    input_size = int(input_size) // VGGT_PATCH_SIZE * VGGT_PATCH_SIZE
    if keep_aspect:
        out_w = input_size
        out_h = max(VGGT_PATCH_SIZE, round(input_size * h / w / VGGT_PATCH_SIZE) * VGGT_PATCH_SIZE)
    else:
        out_h = out_w = input_size
    frames = F.interpolate(frames, size=(out_h, out_w), mode="bilinear", align_corners=False, antialias=True)
    return frames.clamp(0.0, 1.0)


@torch.no_grad()
def get_vggt_target(vggt_model, pil_images, future_pil_images, img_masks, vggt_cfg):
    """Run frozen VGGT jointly over the configured views at t (and t+T).

    Args:
        pil_images / future_pil_images: (B, N_cam, 3, H, W) uint8 RGB.
        img_masks: (B, N_cam) bool, False for zero-padded missing cameras.

    Returns:
        dict with
          current / future: (B, V*G*G, C) float32 patch features pooled to a GxG grid per view,
          current_mask / future_mask: (B, V*G*G) bool token validity.
        `future*` entries are None when vggt.use_future is False.
    """
    views = vggt_views(vggt_cfg)
    grid = int(vggt_cfg.get("grid_size", 8))
    use_future = bool(vggt_cfg.get("use_future", True))
    input_size = int(vggt_cfg.get("input_size", 518))
    keep_aspect = bool(vggt_cfg.get("keep_aspect", True))

    frames = [pil_images[:, views]]
    if use_future:
        if future_pil_images is None:
            raise ValueError("align_params.vggt.use_future=True requires future_pil_images (data.use_future_image=True).")
        frames.append(future_pil_images[:, views])
    frames = torch.cat(frames, dim=1)  # (B, S, 3, H, W) with S = T * V
    bsize, seq_len = frames.shape[:2]
    num_time = 2 if use_future else 1
    num_views = len(views)

    frames = frames.flatten(0, 1).float() / 255.0
    frames = _resize_for_vggt(frames, input_size, keep_aspect)
    frames = frames.view(bsize, seq_len, *frames.shape[1:])
    patch_h = frames.shape[-2] // VGGT_PATCH_SIZE
    patch_w = frames.shape[-1] // VGGT_PATCH_SIZE

    if img_masks is None:
        view_valid = torch.ones(bsize, num_views, dtype=torch.bool, device=frames.device)
    else:
        view_valid = img_masks[:, views].to(device=frames.device, dtype=torch.bool)
    frame_valid = view_valid.repeat(1, num_time)  # (B, S)

    feat_dim = 2 * VGGT_EMBED_DIM
    targets = torch.zeros(bsize, seq_len, grid * grid, feat_dim, device=frames.device, dtype=torch.float32)
    # Missing cameras are dropped from VGGT's input (a black frame would still take
    # part in global attention), so samples are grouped by their camera-validity pattern.
    for pattern in torch.unique(frame_valid.to(torch.uint8), dim=0).bool():
        frame_idx = pattern.nonzero(as_tuple=True)[0]
        if frame_idx.numel() == 0:
            continue
        sample_idx = (frame_valid == pattern).all(dim=1).nonzero(as_tuple=True)[0]
        sub = frames[sample_idx][:, frame_idx]
        output_list, patch_start_idx = vggt_model(sub)
        feats = output_list[vggt_model.vggt_layer][:, :, patch_start_idx:]  # (b, s, P, 2C)
        b, s = feats.shape[:2]
        feats = feats.float().reshape(b * s, patch_h, patch_w, feat_dim).permute(0, 3, 1, 2)
        feats = F.adaptive_avg_pool2d(feats, grid)
        feats = feats.flatten(2).transpose(1, 2).reshape(b, s, grid * grid, feat_dim)
        targets[sample_idx[:, None], frame_idx[None, :]] = feats

    token_mask = frame_valid[:, :, None].expand(-1, -1, grid * grid)
    tokens_per_step = num_views * grid * grid
    targets = targets.reshape(bsize, num_time * tokens_per_step, feat_dim)
    token_mask = token_mask.reshape(bsize, num_time * tokens_per_step)
    return {
        "current": targets[:, :tokens_per_step],
        "current_mask": token_mask[:, :tokens_per_step],
        "future": targets[:, tokens_per_step:] if use_future else None,
        "future_mask": token_mask[:, tokens_per_step:] if use_future else None,
    }
