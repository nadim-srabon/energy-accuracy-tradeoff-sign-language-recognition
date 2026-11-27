"""Prefetch all pretrained weights used by the five notebooks into local caches.

Must run OUTSIDE any measured energy window (per gpu_package README). The exact
identifiers were extracted from the notebook sources:
  mobilenet : torchvision MobileNet_V3_Small_Weights.IMAGENET1K_V1
  resnet    : torchvision ResNet50_Weights.IMAGENET1K_V2
  vivit     : timm vit_tiny_patch16_224
  timesformer : HF facebook/timesformer-base-finetuned-k400
                (AutoImageProcessor + TimesformerModel)
  hybrid    : torchvision ConvNeXt_Tiny + Swin_T (IMAGENET1K_V1)
"""
import torch
import torchvision
import timm
from transformers import AutoImageProcessor, TimesformerModel

torch.hub.set_dir(torch.hub.get_dir())  # default caches


def tv(name, ctor, weights):
    m = ctor(weights=weights)
    n = sum(p.numel() for p in m.parameters())
    print(f"{name}: ok ({n/1e6:.2f}M params)")


tv("mobilenet_v3_small", torchvision.models.mobilenet_v3_small,
   torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
tv("resnet50", torchvision.models.resnet50, torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
tv("convnext_tiny", torchvision.models.convnext_tiny,
   torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
tv("swin_t", torchvision.models.swin_t, torchvision.models.Swin_T_Weights.IMAGENET1K_V1)

m = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0)
print(f"timm vit_tiny_patch16_224: ok ({sum(p.numel() for p in m.parameters())/1e6:.2f}M params)")

proc = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
tm = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")
print(f"timesformer: ok (mean={proc.image_mean}, std={proc.image_std}, "
      f"params={sum(p.numel() for p in tm.parameters())/1e6:.2f}M)")
print("All pretrained weights cached.")
