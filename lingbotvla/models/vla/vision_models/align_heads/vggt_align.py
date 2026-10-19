import torch
import torch.nn as nn

from .depth_head import TaskTokenDepthHead


class VGGTAlignModule(nn.Module):
    """Training-only student heads for multi-view VGGT relational distillation.

    The current head reads [image tokens of the configured views, current query]
    and the future head reads [image tokens, future query]; each predicts one
    token per pooled VGGT patch (V * G * G). Nothing here is used by
    `sample_actions`, so the inference graph is unchanged.
    """

    def __init__(self, vggt_cfg, llm_hidden_size):
        super().__init__()
        self.views = [int(v) for v in vggt_cfg.get("views", [0, 1, 2])]
        self.grid = int(vggt_cfg.get("grid_size", 8))
        self.tokens_per_view = self.grid * self.grid
        self.num_tokens = len(self.views) * self.tokens_per_view
        self.use_future = bool(vggt_cfg.get("use_future", True))
        head_cfg = {
            "num_backbone_tokens": self.num_tokens,
            "dim_out": int(vggt_cfg.get("dim_out", 1024)),
            "num_layers": int(vggt_cfg.get("num_layers", 1)),
            "num_heads": int(vggt_cfg.get("num_heads", 4)),
            "dim_head": int(vggt_cfg.get("dim_head", 32)),
            "ff_mult": int(vggt_cfg.get("ff_mult", 1)),
        }

        self.current_queries = nn.Parameter(torch.randn(self.num_tokens, llm_hidden_size))
        self.current_head = TaskTokenDepthHead(head_cfg, llm_hidden_size=llm_hidden_size).to(dtype=torch.bfloat16)
        if self.use_future:
            self.future_queries = nn.Parameter(torch.randn(self.num_tokens, llm_hidden_size))
            self.future_head = TaskTokenDepthHead(head_cfg, llm_hidden_size=llm_hidden_size).to(dtype=torch.bfloat16)

    def _reset_post_init_parameters(self):
        # Called from FlowMatching._init_weights when these params are missing from a
        # pretrained checkpoint; Linear/LayerNorm sub-layers are handled there.
        nn.init.normal_(self.current_queries, std=1.0)
        if self.use_future:
            nn.init.normal_(self.future_queries, std=1.0)

    @staticmethod
    def _predict(head, queries, context):
        queries = queries.repeat(context.shape[0], 1, 1).to(dtype=context.dtype, device=context.device)
        return head(context, queries).contiguous().float()

    def forward(self, current_context, future_context=None):
        current_preds = self._predict(self.current_head, self.current_queries, current_context)
        future_preds = None
        if self.use_future and future_context is not None:
            future_preds = self._predict(self.future_head, self.future_queries, future_context)
        return current_preds, future_preds
