"""Relational feature distillation (MECo-WAM Eq. 7, 10, 11, 14).

Student and teacher features are compared through their mean-normalized
pairwise L2 distance matrices, so the two need not share a feature space or
dimension. The temporal term matches how those relations change between the
current and the future frame.
"""
import torch


def pairwise_relations(feats, token_mask, clip=100.0, eps=1e-6):
    """Mean-normalized pairwise L2 distances.

    Args:
        feats: (B, N, D) features.
        token_mask: (B, N) bool, True for valid tokens.

    Returns:
        relations: (B, N, N) float32, divided by the mean valid off-diagonal distance and clipped.
        pair_mask: (B, N, N) bool, valid off-diagonal pairs.
    """
    feats = feats.float()
    sq_norm = feats.pow(2).sum(dim=-1)
    dist_sq = sq_norm[:, :, None] + sq_norm[:, None, :] - 2.0 * feats @ feats.transpose(1, 2)
    # clamp before sqrt keeps the zero diagonal differentiable
    dist = dist_sq.clamp_min(eps).sqrt()

    num_tokens = feats.shape[1]
    eye = torch.eye(num_tokens, dtype=torch.bool, device=feats.device)
    token_mask = token_mask.to(device=feats.device, dtype=torch.bool)
    pair_mask = token_mask[:, :, None] & token_mask[:, None, :] & ~eye

    pair_weight = pair_mask.float()
    mean_dist = (dist * pair_weight).sum(dim=(1, 2)) / pair_weight.sum(dim=(1, 2)).clamp_min(1.0)
    relations = dist / mean_dist.clamp_min(eps)[:, None, None]
    return relations.clamp(max=clip), pair_mask


def masked_mean(values, mask):
    weight = mask.float()
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def relation_loss(student_rel, teacher_rel, pair_mask):
    return masked_mean((student_rel - teacher_rel).abs(), pair_mask)


def relation_change(rel_from, rel_to, eps=1e-6):
    return (rel_to - rel_from) / (rel_to.abs() + rel_from.abs() + eps)


def temporal_relation_loss(student_cur, student_fut, teacher_cur, teacher_fut, pair_mask):
    diff = relation_change(student_cur, student_fut) - relation_change(teacher_cur, teacher_fut)
    return masked_mean(diff.abs(), pair_mask)


def same_view_pair_mask(num_views, tokens_per_view, device):
    """(N, N) bool, True for token pairs from the same camera view."""
    view_id = torch.arange(num_views * tokens_per_view, device=device) // tokens_per_view
    return view_id[:, None] == view_id[None, :]
