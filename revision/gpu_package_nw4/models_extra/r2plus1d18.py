# Architecture cell for the extra CNN-family model. Executed by run_ext.py in the
# namespace of the ResNet-50 notebook (same CONFIG, data pipeline, training loop,
# early stopping and energy tracking); only this cell differs.
#
# R(2+1)D-18 (Tran et al., CVPR 2018), torchvision weights KINETICS400_V1.
# Why it was added: it is a video-pretrained CNN (Kinetics-400), like TimeSformer
# on the Transformer side, and of similar size to Video Swin-T (the extra
# Transformer). This separates "CNN vs Transformer" from "image vs video
# pretraining", the confound raised by reviewers R3.3 and R4.1.
#
# Kept identical to the ResNet-50 notebook: 16 frames at 224x224, partial freeze
# up to CONFIG.UNFREEZE_FROM_LAYER ("layer4" trainable), dropout 0.3, linear head.
# Input normalisation is the weights' own Kinetics mean/std (set by run_ext.py).


class R2Plus1D18VideoCNN(nn.Module):
    """CNN family, video-pretrained member: factorised (2+1)D convolutions over the
    whole clip, global spatio-temporal average pooling, linear classification head."""

    def __init__(self, num_classes, unfreeze_from_layer=CONFIG.UNFREEZE_FROM_LAYER):
        super().__init__()
        weights = torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1
        backbone = torchvision.models.video.r2plus1d_18(weights=weights)
        self.out_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()              # keep the pooled clip feature
        self.backbone = backbone

        # Same partial-freeze rule as the ResNet-50 notebook: freeze every stage
        # before `unfreeze_from_layer`, train that stage and everything after it.
        unfreeze = False
        for name, module in self.backbone.named_children():
            if name == unfreeze_from_layer:
                unfreeze = True
            for p in module.parameters():
                p.requires_grad = unfreeze
        if not any(p.requires_grad for p in self.backbone.parameters()):
            raise ValueError(f'No backbone stage named {unfreeze_from_layer!r}')

        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(self.out_dim, num_classes)

    def forward(self, x):
        # x: (B, T, 3, H, W) from the shared dataset -> (B, 3, T, H, W) for 3D convs
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        feats = self.backbone(x)        # (B, 512)
        feats = self.dropout(feats)
        return self.head(feats)


model = R2Plus1D18VideoCNN(num_classes=NUM_CLASSES).to(CONFIG.DEVICE)
n_params_total = sum(p.numel() for p in model.parameters())
n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {n_params_total:,}")
print(f"Trainable parameters: {n_params_trainable:,}")
