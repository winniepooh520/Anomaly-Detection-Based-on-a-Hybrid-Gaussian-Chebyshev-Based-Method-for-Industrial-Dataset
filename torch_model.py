"""PyTorch model for the PaDiM model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from random import sample
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from anomalib.models.components import FeatureExtractor, MultiVariateGaussian
from anomalib.models.padim.anomaly_map import AnomalyMapGenerator
from anomalib.pre_processing import Tiler

DIMS = {
    "resnet18": {"orig_dims": 448, "reduced_dims": 100, "emb_scale": 4},
    "wide_resnet50_2": {"orig_dims": 1792, "reduced_dims": 550, "emb_scale": 4},
}


class PadimModel(nn.Module):
    """Padim Module.

    Args:
        input_size (Tuple[int, int]): Input size for the model.
        layers (List[str]): Layers used for feature extraction
        backbone (str, optional): Pre-trained model backbone. Defaults to "resnet18".
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        layers: List[str],
        backbone: str = "resnet18",
        pre_trained: bool = True,
    ):
        super().__init__()
        self.tiler: Optional[Tiler] = None

        self.backbone = backbone
        self.layers = layers
        self.feature_extractor = FeatureExtractor(backbone=self.backbone, layers=layers, pre_trained=pre_trained)
        self.dims = DIMS[backbone]
        # pylint: disable=not-callable
        # Since idx is randomly selected, save it with model to get same results
        self.register_buffer(
            "idx",
            torch.tensor(sample(range(0, DIMS[backbone]["orig_dims"]), DIMS[backbone]["reduced_dims"])),
        )
        self.idx: Tensor
        self.loss = None
        self.anomaly_map_generator = AnomalyMapGenerator(image_size=input_size)

        n_features = DIMS[backbone]["reduced_dims"]
        patches_dims = torch.tensor(input_size) / DIMS[backbone]["emb_scale"]
        n_patches = patches_dims.ceil().prod().int().item()
        self.gaussian = MultiVariateGaussian(n_features, n_patches)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Forward-pass image-batch (N, C, H, W) into model to extract features.

        Args:
            input_tensor: Image-batch (N, C, H, W)
            input_tensor: Tensor:

        Returns:
            Features from single/multiple layers.

        Example:
            >>> x = torch.randn(32, 3, 224, 224)
            >>> features = self.extract_features(input_tensor)
            >>> features.keys()
            dict_keys(['layer1', 'layer2', 'layer3'])

            >>> [v.shape for v in features.values()]
            [torch.Size([32, 64, 56, 56]),
            torch.Size([32, 128, 28, 28]),
            torch.Size([32, 256, 14, 14])]
        """

        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            embeddings = self.generate_embedding(features)

        if self.tiler:
            embeddings = self.tiler.untile(embeddings)

        if self.training:
            output = embeddings
        else:
            output = self.anomaly_map_generator(
                embedding=embeddings, mean=self.gaussian.mean, std=self.gaussian.std, inv_covariance=self.gaussian.inv_covariance
            )
        return output

    # def generate_embedding(self, features: Dict[str, Tensor]) -> Tensor:
    #     """Generate embedding from hierarchical feature map.

    #     Args:
    #         features (Dict[str, Tensor]): Hierarchical feature map from a CNN (ResNet18 or WideResnet)
        
    #     after interpolate:
    #     layer0:  torch.Size([32,  64, 56, 56])
    #     layer1:  torch.Size([32, 128, 56, 56])
    #     layer2:  torch.Size([32, 256, 56, 56])
    #     layer3:  torch.Size([32, 512, 56, 56])
        
    #     repeat channel:
    #     layer0 + layer1: torch.Size([32, 128, 56, 56]) + torch.Size([32, 128, 56, 56])
    #     layer1 + layer2: torch.Size([32, 256, 56, 56]) + torch.Size([32, 256, 56, 56])
    #     layer2 + layer3: torch.Size([32, 512, 56, 56]) + torch.Size([32, 512, 56, 56])
        
    #     embeddings = cat[layer0 + layer1, layer1 + layer2, layer2 + layer3]

    #     Returns:
    #         Embedding vector
    #     """

    #     embeddings = features[self.layers[0]] #  #32,64,56,56
    #     b,c,h,w = features[self.layers[0]].shape
        
    #     for layer in self.layers[1:]:
    #         layer_embedding = features[layer]
    #         layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            
    #         if layer == self.layers[1]:
    #             channel_scale = int(features[layer].shape[1]/c)   #   channel_scale =  layer1_channel(128) /layer0_channel(64) = 2        
    #             temp_layer = features[self.layers[0]].repeat(1, channel_scale, 1, 1) # layer0: ([32, 64, 56, 56])→ ([32, 128, 56, 56])
    #             layer_diff = layer_embedding+temp_layer # layer 1 - layer 0
    #             temp_layer = layer_embedding
    #         else:
    #             channel_scale = int(features[layer].shape[1]/temp_layer.shape[1])          
    #             temp_layer = temp_layer.repeat(1, channel_scale, 1, 1)
    #             layer_diff = layer_embedding+temp_layer
    #             temp_layer = layer_embedding
    #         embeddings = torch.cat((embeddings, layer_diff), 1)
            

    #     # # subsample embeddings
    #     idx = self.idx.to(embeddings.device)
    #     embeddings = torch.index_select(embeddings, 1, idx)
    #     return embeddings
    
    
    def generate_embedding(self, features: Dict[str, Tensor]) -> Tensor:
        """Generate embedding from hierarchical feature map.

        Args:
            features (dict[str, Tensor]): Hierarchical feature map from a CNN (ResNet18 or WideResnet)

        Returns:
            Embedding vector
        """

        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        # subsample embeddings
        idx = self.idx.to(embeddings.device)
        embeddings = torch.index_select(embeddings, 1, idx)
        return embeddings
