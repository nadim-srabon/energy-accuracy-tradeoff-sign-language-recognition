# Architecture cell for the extra Transformer-family model. Executed by run_ext.py
# in the namespace of the ViViT-Lite notebook (same CONFIG, 16-frame data pipeline,
# training loop, early stopping and energy tracking); only this cell differs.
#
# Video Swin Transformer, tiny (Liu et al., CVPR 2022), torchvision weights
# KINETICS400_V1. Why it was added: a video-pretrained Transformer of similar size
# to R(2+1)D-18 (the extra CNN), pretrained on the same data, so the pair gives a
# CNN-vs-Transformer comparison with pretraining and capacity held roughly equal
# (reviewers R3.3, R4.1). It also uses shifted-window space-time attention, a
# third attention design next to ViViT-Lite (factorised) and TimeSformer (divided).
#
# Partial freeze: only the last top-level stage of `features` and the final norm
# are trainable -- the same rule the hybrid notebook applies to its Swin-T branch.
# 16 frames at 224x224, dropout 0.3, linear head. Input normalisation is the
# weights' own mean/std (set by run_ext.py).

SWIN3D_UNFROZEN_STAGES = 1


class Swin3DTinyVideoTransformer(nn.Module):
    """Transformer family, video-pretrained mid-size member: shifted-window
    space-time self-attention over the whole clip, pooled feature, linear head."""

    def __init__(self, num_classes, unfrozen_stages=SWIN3D_UNFROZEN_STAGES):
        super().__init__()
        weights = torchvision.models.video.Swin3D_T_Weights.KINETICS400_V1
        backbone = torchvision.models.video.swin3d_t(weights=weights)
        self.out_dim = backbone.head.in_features  # 768
        backbone.head = nn.Identity()             # keep the pooled clip feature
        self.backbone = backbone

        for p in self.backbone.parameters():
            p.requires_grad = False
        top_level_stages = list(self.backbone.features.children())
        n_trainable = min(unfrozen_stages, len(top_level_stages))
        for stage in top_level_stages[len(top_level_stages) - n_trainable:]:
            for p in stage.parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = True

        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(self.out_dim, num_classes)

    def forward(self, x):
        # x: (B, T, 3, H, W) from the shared dataset -> (B, 3, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        feats = self.backbone(x)        # (B, 768)
        feats = self.dropout(feats)
        return self.head(feats)


model = Swin3DTinyVideoTransformer(num_classes=NUM_CLASSES).to(CONFIG.DEVICE)
n_params_total = sum(p.numel() for p in model.parameters())
n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {n_params_total:,}")
print(f"Trainable parameters: {n_params_trainable:,}")
