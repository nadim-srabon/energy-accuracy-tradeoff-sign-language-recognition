"""Download / verify every pretrained weight used by the 8 models before any GPU job.

Runs outside any measured window. Old models: same identities as
revision/local_setup/prefetch_weights.py. New models: R(2+1)D-18 and Swin3D-T
(torchvision, Kinetics-400) and MobileViT-S (timm, ImageNet-1k).
"""
import torch
import torchvision
import timm
from transformers import AutoImageProcessor, TimesformerModel


def tv(name, ctor, weights):
    m = ctor(weights=weights)
    print(f'{name}: ok ({sum(p.numel() for p in m.parameters())/1e6:.2f}M params)')


tv('mobilenet_v3_small', torchvision.models.mobilenet_v3_small,
   torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
tv('resnet50', torchvision.models.resnet50, torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
tv('convnext_tiny', torchvision.models.convnext_tiny, torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
tv('swin_t', torchvision.models.swin_t, torchvision.models.Swin_T_Weights.IMAGENET1K_V1)
m = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
print(f'timm vit_tiny_patch16_224: ok ({sum(p.numel() for p in m.parameters())/1e6:.2f}M params)')
AutoImageProcessor.from_pretrained('facebook/timesformer-base-finetuned-k400')
tm = TimesformerModel.from_pretrained('facebook/timesformer-base-finetuned-k400')
print(f'timesformer: ok ({sum(p.numel() for p in tm.parameters())/1e6:.2f}M params)')

# New models
tv('r2plus1d_18 (K400)', torchvision.models.video.r2plus1d_18,
   torchvision.models.video.R2Plus1D_18_Weights.KINETICS400_V1)
tv('swin3d_t (K400)', torchvision.models.video.swin3d_t,
   torchvision.models.video.Swin3D_T_Weights.KINETICS400_V1)
m = timm.create_model('mobilevit_s.cvnets_in1k', pretrained=True, num_classes=0)
print(f'timm mobilevit_s.cvnets_in1k: ok ({sum(p.numel() for p in m.parameters())/1e6:.2f}M params)')
print('All pretrained weights cached. torch', torch.__version__)
