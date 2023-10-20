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

# New 降維 manifold.TSNE
#from sklearn import manifold

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
        super().__init__() # super: 繼承父類別(nn.module)的參數
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
            
        After F.interpolate:
            [layer.shape for layers in features.values()]
            torch.Size([3, 64, 56, 56])
            torch.Size([3, 128, 56, 56])
            torch.Size([3, 256, 56, 56])
        
        """

        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)

        with torch.no_grad():
            features = self.feature_extractor(input_tensor) #features 是dic{} dict_keys(['layer1', 'layer2', 'layer3'])
            embeddings = self.generate_embedding(features) #embedding 是將dic的key(['layer1', 'layer2', 'layer3'])都去掉，只留下其內tensor並embedding，長度不足者並排[(0,1)->(0,0,1,1)]，並且隨機壓縮過的結果。123

        if self.tiler:
            embeddings = self.tiler.untile(embeddings)

        if self.training:
            output = embeddings # embedding結束 = training結束
        else: # testing階段則要產生 anomaly_map
            output = self.anomaly_map_generator(
                embedding=embeddings, mean=self.gaussian.mean, inv_covariance=self.gaussian.inv_covariance
            )
        return output

    def generate_embedding(self, features: Dict[str, Tensor]) -> Tensor:
        """Generate embedding from hierarchical feature map.

        Args:
            features (Dict[str, Tensor]): Hierarchical feature map from a CNN (ResNet18 or WideResnet)

        Returns:
            Embedding vector
        """

        embeddings = features[self.layers[0]]
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest") # embeddings.shape[-2:] = [56,56]
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        # subsample embeddings
        idx = self.idx.to(embeddings.device)
        embeddings = torch.index_select(embeddings, 1, idx)
        return embeddings
    
        # # New TSNE subsample embeddings
        # b,C,H,W = embeddings.shape #torch.Size([32, 960, 56, 56])
        # embeddings = embeddings.reshape(b*C,H*W) # tsne 只能吃2維資料，所以先轉成2維，TSNE完再轉回4維
        # embeddings = embeddings.detach().numpy() #tsne 只能吃numpy資料，所以tsne完再轉回tensor
        # X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(embeddings)
        # X_tsne = X_tsne.reshape(b,C,2,1) # 1+1 = 2 = 上一行的n_components,1只是維度從(b,c,2)->((b,c,2))
        # X_tsne = torch.from_numpy(X_tsne)  # 將X_tsne.numpy 轉成tensor的形式回傳
        # return  X_tsne
        # ## New end
