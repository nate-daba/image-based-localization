from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNet(nn.Module):
    """Defines Siamese network structure made up of two identical feature exractors.
    Args:
        ground_net (nn.Module): feature extractor for the ground view images.
        aerial_net (nn.Module): feature extractor for the aerial view images.
    
    """
    def __init__(self, 
                 ground_net: nn.Module=None, 
                 aerial_net: nn.Module=None)->None:
        super(SiameseNet, self).__init__()
        
        self.ground_feature_extractor = ground_net
        self.aerial_feature_extractor = aerial_net
        
        self.fc_layer = nn.Linear(512, 4096, bias=True)
        
    def forward(self, 
                ground_image: Tensor, 
                aerial_image: Tensor)->Tuple[Tensor]:
        """Extracts features from ground and aerial view images. Passes features through 
        linear layer. Computes L2 norms of linear layer outputs to compute final embedding.
        
        Args:
            ground_image (Tensor): ground view image.
            aerial_image (Tensor): aerial view image.
        Returns:
            aerial_global (Tensor): L2 normalized embedding of aerial image.
            ground_global (Tensor): L2 normalized embedding of ground image.
            aerial_features (Tensor): extracted features of aerial image.
            ground_features (Tensor): extracted features of ground image.
            aerial_fc (Tensor): output of linear layer for aerial image.
            ground_fc (Tensor): output of linear layer for ground image.
            
        """
        # extracted features: (B, C, H, W)
        ground_features = self.ground_feature_extractor(ground_image)
        aerial_features = self.aerial_feature_extractor(aerial_image)
        
        # global average pooling (gap): (B, C)
        ground_gap = torch.mean(ground_features, (2, 3))
        aerial_gap = torch.mean(aerial_features, (2, 3))
        
        # pass to fully connected (fc) layer: (B, 4096)
        ground_fc = self.fc_layer(ground_gap)
        aerial_fc = self.fc_layer(aerial_gap)
        
        # L2 normalize: (B, 4096)
        ground_global = F.normalize(ground_fc, p=2, dim=1)
        aerial_global = F.normalize(aerial_fc, p=2, dim=1)
        
        return (aerial_global, ground_global, aerial_features, 
                ground_features, aerial_fc, ground_fc)