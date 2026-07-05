"""Backbone + classification head via `timm`."""

import timm
import torch.nn as nn


DEFAULT_BACKBONE = "resnet50"


def build_model(num_classes: int, backbone: str = DEFAULT_BACKBONE, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model
