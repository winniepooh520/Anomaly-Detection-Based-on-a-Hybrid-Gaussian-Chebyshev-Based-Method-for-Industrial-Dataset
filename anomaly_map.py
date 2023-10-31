"""Anomaly Map Generator for the PaDiM model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from torch import Tensor, nn

from anomalib.models.components import GaussianBlur2d


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap.

    Args:
        image_size (Union[ListConfig, Tuple]): Size of the input image. The anomaly map is upsampled to this dimension.
        sigma (int, optional): Standard deviation for Gaussian Kernel. Defaults to 4.
    """

    def __init__(self, image_size: Union[ListConfig, Tuple], sigma: int = 4):
        super().__init__()
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)

    @staticmethod
    
    # # Hank: centerlize ver 2 , multi_variate_gaussian.py/def forward
    # def compute_distance(embedding: Tensor, stats: List[Tensor]) -> Tensor:
    #     """Compute anomaly score to the patch in position(i,j) of a test image.

    #     Ref: Equation (2), Section III-C of the paper.

    #     Args:
    #         embedding (Tensor): Embedding Vector
    #         stats (List[Tensor]): Mean and std Matrix of the multivariate Gaussian distribution

    #     Returns:
    #         Anomaly score of a test image via mahalanobis distance.
    #     """

    #     batch, channel, height, width = embedding.shape #32,960,56,56       
    #     embedding = embedding.reshape(-1)# shape:[96337920] Hank: centerlize ver 2 , multi_variate_gaussian.py/def forward 

    #     # calculate Chebyshev distances
    #     mean, std = stats # mean of batch dimention
    #     # mean = mean.unsqueeze(-1) # V1
    #     # std = std.unsqueeze(-1) # V1
    #     # # V2 do nothing
    #     # # V2 do nothing
       
    #     delta = torch.abs(embedding - mean)
        
    #     #k = torch.nan_to_num(delta/std)
    #     k = torch.nan_to_num(delta/std)
    #     distances = (((k)**2)-1)/((k)**2)# 1-distance
    #     distances = distances.reshape(batch, channel, height, width) # distance.shape:[96337920] = 32*960*56*56
    #     distances = torch.max(distances, 1)[0]
    #     distances = distances.reshape(batch, 1, height, width)
        
    #     return distances
        



    
    # #Hank: centerlize ver 1 , multi_variate_gaussian.py/def forward
    # def compute_distance(embedding: Tensor, stats: List[Tensor]) -> Tensor:
    #     """Compute anomaly score to the patch in position(i,j) of a test image.

    #     Ref: Equation (2), Section III-C of the paper.

    #     Args:
    #         embedding (Tensor): Embedding Vector
    #         stats (List[Tensor]): Mean and std Matrix of the multivariate Gaussian distribution

    #     Returns:
    #         Anomaly score of a test image via mahalanobis distance.
    #     """

    #     batch, channel, height, width = embedding.shape
        
    #     embedding = embedding.reshape(-1, channel)# Hank: centerlize ver 1 , multi_variate_gaussian.py/def forward
    #     #embedding = embedding.reshape(-1)# Hank: centerlize ver 2 , multi_variate_gaussian.py/def forward

    #     # calculate Chebyshev distances
    #     mean, std = stats # mean of batch dimention
    #     # mean = mean.unsqueeze(-1) # V1
    #     # std = std.unsqueeze(-1) # V1
    #     # # V2 do nothing
    #     # # V2 do nothing
    #     mean = torch.mean(mean, dim=1) # # mean of height*width dimention
    #     std = torch.std(std, dim=1)
        
    #     delta = (embedding - mean)
        
    #     k = torch.nan_to_num(delta/std)
    #     distances = (((k)**2)-1)/((k)**2)# 1-distance
    #     distances = torch.nan_to_num(distances, neginf=0)
    #     distances = torch.max(distances, 1)[0]
    #     # distances = torch.max(distances)[0] # Hank: centerlize ver 2
    #     distances = distances.reshape(batch, 1, height, width)

    #     return distances
    
    
    # # simple Chebyshev
    # def compute_distance(embedding: Tensor, stats: List[Tensor]) -> Tensor:
    #     """Compute anomaly score to the patch in position(i,j) of a test image.

    #     Ref: Equation (2), Section III-C of the paper.

    #     Args:
    #         embedding (Tensor): Embedding Vector
    #         stats (List[Tensor]): Mean and std Matrix 

    #     Returns:
    #         Anomaly score of a test image via Chebyshev distance.
    #     """
    #     """
    #     抽projection 
    #     cos similarity (average)
    #     1st chebyshev prob: (abs(normal max - normal mean)/normal std) => normal k =>
    #     1/ normal k**2 = normal prob => 
    #     total feature number * normal prob (ex: normal prob = 0.8, total feature number = 1000 => projection number = 1000 * 0.8 = 800)
    #     """
    #     batch, channel, height, width = embedding.shape
    #     embedding = embedding.reshape(batch, channel, height * width)

    #     # calculate Chebyshev distances
    #     mean, std, inv_covariance = stats
        
    #     diff = (embedding - mean)
        
    #     k = torch.nan_to_num(diff/std)
    #     cheby_distances = ((k**2)-1)/(k**2)# 1-(1/(k^2))
    #     cheby_distances = torch.nan_to_num(cheby_distances)
        
    #     cheby_distances = torch.max(cheby_distances, 1)[0]
    #     cheby_distances = cheby_distances.reshape(batch, 1, height, width)
    #     # cheby_distances = (torch.max(cheby_distances, 1)[0])*0.75 # k scale down
    #     # cheby_distances = cheby_distances.reshape(batch, 1, height, width)
        
    #     return cheby_distances
    
    
    
    
    
    # Gaussian_distances+cheby_distances)/2
    def compute_distance(embedding: Tensor, stats: List[Tensor]) -> Tensor:
        """Compute anomaly score to the patch in position(i,j) of a test image.

        Ref: Equation (2), Section III-C of the paper.

        Args:
            embedding (Tensor): Embedding Vector
            stats (List[Tensor]): Mean and Covariance Matrix of the multivariate Gaussian distribution

        Returns:
            Anomaly score of a test image via mahalanobis distance.
        """

        batch, channel, height, width = embedding.shape
        embedding = embedding.reshape(batch, channel, height * width)

        
        mean, std, inv_covariance = stats
        
        # calculate Chebyshev distances       
        diff = (embedding - mean)
        
        k = torch.nan_to_num(diff/std)
        cheby_distances = ((k**2)-1)/(k**2)# 1-(1/(k^2))
        cheby_distances = torch.nan_to_num(cheby_distances)
        
        cheby_distances = torch.max(cheby_distances, 1)[0]
        cheby_distances = cheby_distances.reshape(batch, 1, height, width)
        # cheby_distances = (torch.max(cheby_distances, 1)[0])*0.8 # k scale down
        # cheby_distances = cheby_distances.reshape(batch, 1, height, width)
        
        # calculate mahalanobis distances
        delta = (embedding - mean).permute(2, 0, 1)

        Gaussian_distances = (torch.matmul(delta, inv_covariance) * delta).sum(2).permute(1, 0)
        Gaussian_distances = Gaussian_distances.reshape(batch, 1, height, width)
        Gaussian_distances = Gaussian_distances.clamp(0).sqrt()

        return ((Gaussian_distances+cheby_distances)/2)
      
    
    
    
    #         # ver Chebyshev*2
    # def compute_distance(embedding: Tensor, stats: List[Tensor]) -> Tensor:
    #     """Compute anomaly score to the patch in position(i,j) of a test image.

    #     Ref: Equation (2), Section III-C of the paper.

    #     Args:
    #         embedding (Tensor): Embedding Vector
    #         stats (List[Tensor]): Mean and std Matrix 

    #     Returns:
    #         Anomaly score of a test image via Chebyshev distance.
    #     """
    #     """
    #     抽projection 
    #     cos similarity (average)
    #     1st chebyshev prob: (abs(normal max - normal mean)/normal std) => normal k =>
    #     1/ normal k**2 = normal prob => 
    #     total feature number * normal prob (ex: normal prob = 0.8, total feature number = 1000 => projection number = 1000 * 0.8 = 800)
    #     """
    #     batch, channel, height, width = embedding.shape
    #     embedding = embedding.reshape(batch, channel, height * width)

    #     # calculate Chebyshev 1st
    #     mean, std = stats
    #     delta = (embedding - mean)
    #     k = torch.nan_to_num(delta/std)
    #     distances = (((k)**2)-1)/((k)**2)# 1-(1/(k^2))
    #     distances = torch.nan_to_num(distances) # distances.shape:([32, 960, 3136])
    #     Channel_max, _ = torch.max(distances, 1)#[0]　# Channel_max.shape:([32, 3136])
    #     Batch_max, _ = torch.max(Channel_max, 0)#[0]　# Batch_max.shape:([3136])
    #     distance_max, _ =  torch.max(Batch_max, 0) #0.9954
        
    #     #　取前embedding.channel*Chebyshev distance.Max個大的 channel維
    #     ChannelScale = int(channel*distance_max)
    #     sorteds, indices = torch.sort(distances, dim=1, descending=True, stable=False, out=None)
    #     embeddingScale = distances[:,:ChannelScale,:] # channel: 960→954
        
    #     # # calculate Chebyshev 2nd
    #     batch2, channel2, hw2 = embeddingScale.shape
    #     mean2 = torch.mean(embeddingScale, dim=0 )
    #     std2 = torch.var(embeddingScale, dim=0)
        
    #     delta = (embeddingScale - mean2)
    #     k = torch.nan_to_num(delta/std2)
    #     distances = (((k)**2)-1)/((k)**2)# 1-(1/(k^2))
    #     distances = torch.nan_to_num(distances) # distances.shape:([32, 960, 3136])
    #     distances, _ = torch.max(distances, 1)
    #     distances = distances.reshape(batch2, 1, height, width)
    #     return distances
    
    
    
    
    # # original PaDiM
    # def compute_distance(embedding: Tensor, stats: List[Tensor]) -> Tensor:
    #     """Compute anomaly score to the patch in position(i,j) of a test image.

    #     Ref: Equation (2), Section III-C of the paper.

    #     Args:
    #         embedding (Tensor): Embedding Vector
    #         stats (List[Tensor]): Mean and Covariance Matrix of the multivariate Gaussian distribution

    #     Returns:
    #         Anomaly score of a test image via mahalanobis distance.
    #     """

    #     batch, channel, height, width = embedding.shape
    #     embedding = embedding.reshape(batch, channel, height * width)

    #     # calculate mahalanobis distances
    #     mean, std, inv_covariance = stats
    #     delta = (embedding - mean).permute(2, 0, 1)

    #     distances = (torch.matmul(delta, inv_covariance) * delta).sum(2).permute(1, 0)
    #     distances = distances.reshape(batch, 1, height, width)
    #     distances = distances.clamp(0).sqrt()

    #     return distances

    def up_sample(self, distance: Tensor) -> Tensor:
        """Up sample anomaly score to match the input image size.

        Args:
            distance (Tensor): Anomaly score computed via the mahalanobis distance.

        Returns:
            Resized distance matrix matching the input image size
        """

        score_map = F.interpolate(
            distance,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )
        return score_map

    def smooth_anomaly_map(self, anomaly_map: Tensor) -> Tensor:
        """Apply gaussian smoothing to the anomaly map.

        Args:
            anomaly_map (Tensor): Anomaly score for the test image(s).

        Returns:
            Filtered anomaly scores
        """

        blurred_anomaly_map = self.blur(anomaly_map)
        return blurred_anomaly_map

    def compute_anomaly_map(self, embedding: Tensor, mean: Tensor, std: Tensor, inv_covariance: Tensor) -> Tensor:
        """Compute anomaly score.

        Scores are calculated based on embedding vector, mean and covariance of the multivariate gaussian
        distribution.

        Args:
            embedding (Tensor): Embedding vector extracted from the test set.
            mean (Tensor): Mean of the multivariate gaussian distribution
            std (Tensor): Inverse Costd matrix of the multivariate gaussian distribution.

        Returns:
            Output anomaly score.
        """

        score_map = self.compute_distance(
            embedding=embedding,
            stats=[mean.to(embedding.device), std.to(embedding.device), inv_covariance.to(embedding.device)],
        )
        up_sampled_score_map = self.up_sample(score_map)
        smoothed_anomaly_map = self.smooth_anomaly_map(up_sampled_score_map)

        return smoothed_anomaly_map

    def forward(self, **kwargs):
        """Returns anomaly_map.

        Expects `embedding`, `mean` and `std` keywords to be passed explicitly.

        Example:
        >>> anomaly_map_generator = AnomalyMapGenerator(image_size=input_size)
        >>> output = anomaly_map_generator(embedding=embedding, mean=mean, std=std)

        Raises:
            ValueError: `embedding`. `mean` or `std` keys are not found

        Returns:
            torch.Tensor: anomaly map
        """

        if not ("embedding" in kwargs and "mean" in kwargs and "std" in kwargs and "inv_covariance" in kwargs):
            raise ValueError(f"Expected keys `embedding`, `mean` and `std` and `covariance` Found {kwargs.keys()}")

        embedding: Tensor = kwargs["embedding"]
        mean: Tensor = kwargs["mean"]
        std: Tensor = kwargs["std"]
        inv_covariance: Tensor = kwargs["inv_covariance"]

        return self.compute_anomaly_map(embedding, mean, std, inv_covariance)
    
    
