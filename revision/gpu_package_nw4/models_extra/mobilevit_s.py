# Architecture cell for the extra Hybrid-family model. Executed by run_ext.py in
# the namespace of the MobileNetV3 notebook (same CONFIG, 16-frame data pipeline,
# training loop, early stopping and energy tracking); only this cell differs.
#
# MobileViT-S (Mehta & Rastegari, ICLR 2022), timm weights
# "mobilevit_s.cvnets_in1k" (ImageNet-1k). Why it was added: the existing hybrid
# is the heaviest model in the study (two ImageNet backbones in parallel); the
# CNN and Transformer families each have a light and a heavy member, the hybrid
# family had only a heavy one. MobileViT-S is a light, single-backbone
# CNN-Transformer hybrid (convolutions plus Transformer blocks inside one
# network) designed for on-device use, so it gives the hybrid family a light
# member and a second hybrid design that does not depend on a fusion gate.
#
# Same template as the CNN notebooks: per-frame backbone, temporal mean pooling,
# dropout 0.3, linear head. Partial freeze: only the last stage and the final
# 1x1 conv are trainable. Frames are fed at the cache's 224x224 (MobileViT was
# pretrained at 256x256; timm resizes internally where a MobileViT block needs
# it). Input normalisation is the weights' own mean/std (set by run_ext.py).

import timm

MOBILEVIT_TIMM_ID = "mobilevit_s.cvnets_in1k"
MOBILEVIT_UNFROZEN_STAGES = 1


class MobileViTSmallStream(nn.Module):
    """Per-frame hybrid embedding extractor (pooled 640-d feature per frame)."""

    def __init__(self, unfrozen_stages=MOBILEVIT_UNFROZEN_STAGES):
        super().__init__()
        self.backbone = timm.create_model(MOBILEVIT_TIMM_ID, pretrained=True, num_classes=0)
        self.out_dim = self.backbone.num_features  # 640

        for p in self.backbone.parameters():
            p.requires_grad = False
        stages = list(self.backbone.stages.children())
        for stage in stages[len(stages) - min(unfrozen_stages, len(stages)):]:
            for p in stage.parameters():
                p.requires_grad = True
        for p in self.backbone.final_conv.parameters():
            p.requires_grad = True

    def forward(self, x):
        # x: (B, T, 3, H, W) -> fold T into batch for the per-frame backbone
        B, T, C, H, W = x.shape
        feats = self.backbone(x.reshape(B * T, C, H, W))   # (B*T, out_dim)
        return feats.view(B, T, self.out_dim)


class HybridMobileViTTemporalAvgModel(nn.Module):
    """Hybrid family, light member: per-frame MobileViT-S embeddings, mean-pooled
    over time, linear classification head (same shape as the CNN notebooks)."""

    def __init__(self, num_classes):
        super().__init__()
        self.hybrid_stream = MobileViTSmallStream()
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(self.hybrid_stream.out_dim, num_classes)

    def forward(self, x):
        feats = self.hybrid_stream(x)   # (B, T, out_dim)
        pooled = feats.mean(dim=1)      # (B, out_dim)
        pooled = self.dropout(pooled)
        return self.head(pooled)


model = HybridMobileViTTemporalAvgModel(num_classes=NUM_CLASSES).to(CONFIG.DEVICE)
n_params_total = sum(p.numel() for p in model.parameters())
n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {n_params_total:,}")
print(f"Trainable parameters: {n_params_trainable:,}")
