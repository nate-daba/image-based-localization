from collections import import collections
from typing import Callable, List, Optional, Tuple, Union


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import models


Tensor = torch.Tensor

class FeatureExtractor(nn.Module):
    """Defines feature extractor model. Uses VGG16 backbone. It directly adopts the first 17 layers of VGG16.
    
    Args:
        initialize_weights (bool): Initialize weights of feature extractor with VGG16 weights from ImageNet-1k.
        
    """
    
    def __init__(self, initialize_weights : Optional[bool] = True) -> None:
        
        super(FeatureExtractor, self).__init__()
            
        self.extract_features = nn.Sequential(OrderedDict([

            # layer 1: conv3-64
            ('conv1_1', nn.Conv2d(3, 64, kernel_size=3, stride=1)),
            ('relu1_1', nn.ReLU(inplace=True)),
            # layer 2: conv3-64
            ('conv1_2', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            ('relu1_2', nn.ReLU(inplace=True)),
            # layer 3: max pooling
            ('layer3_maxpool2x2', nn.MaxPool2d(kernel_size=2, 
                                               stride=2, 
                                               padding=0, 
                                               dilation=1, 
                                               ceil_mode=False)),


            # layer 4: conv3-128
            ('conv2_1', nn.Conv2d(64, 128, kernel_size=3, stride=1)),
            ('relu2_1', nn.ReLU(inplace=True)),
            # layer 5: conv3-128
            ('conv2_2', nn.Conv2d(128, 128, kernel_size=3, stride=1)),
            ('relu2_2', nn.ReLU(inplace=True)),
            # layer 6: max pooling
            ('layer6_maxpool2x2', nn.MaxPool2d(kernel_size=2, 
                                     stride=2, 
                                     padding=0, 
                                     dilation=1, 
                                     ceil_mode=False)), 


            # layer 7: conv3-256
            ('conv3_1', nn.Conv2d(128, 256, kernel_size=3, stride=1)),
            ('relu3_1', nn.ReLU(inplace=True)),
            # layer 8: conv3-256
            ('conv3_2', nn.Conv2d(256, 256, kernel_size=3, stride=1)),
            ('relu3_2', nn.ReLU(inplace=True)),
            # layer 9: conv3-256
            ('conv3_3', nn.Conv2d(256, 256, kernel_size=3, stride=1)),
            ('relu3_3', nn.ReLU(inplace=True)),
            # layer 10: max pooling
            ('layer10_maxpool2x2', nn.MaxPool2d(kernel_size=2, 
                                               stride=2, 
                                               padding=0, 
                                               dilation=1, 
                                               ceil_mode=False)),    


            # layer 11: conv3-512
            ('conv4_1', nn.Conv2d(256, 512, kernel_size=3, stride=1)),
            ('relu4_1', nn.ReLU(inplace=True)),
            # layer 12: conv3-512
            ('conv4_2', nn.Conv2d(512, 512, kernel_size=3, stride=1)),
            ('relu4_2', nn.ReLU(inplace=True)),
            # layer 13: conv3-512
            ('conv4_3', nn.Conv2d(512, 512, kernel_size=3, stride=1)),
            ('relu4_3', nn.ReLU(inplace=True)),
            # layer 14: max pooling
            ('layer14_maxpool2x2', nn.MaxPool2d(kernel_size=2, 
                                               stride=2, 
                                               padding=0, 
                                               dilation=1, 
                                               ceil_mode=False)),


            # layer 15: conv3-512
            ('conv5_1', nn.Conv2d(512, 512, kernel_size=3, stride=1)),
            ('relu5_1', nn.ReLU(inplace=True)),
            # layer 16: conv3-512
            ('conv5_2', nn.Conv2d(512, 512, kernel_size=3, stride=1)),
            ('relu5_2', nn.ReLU(inplace=True)),
            # layer 17: conv3-512
            ('conv5_3', nn.Conv2d(512, 512, kernel_size=3, stride=1)),
            ('relu5_3', nn.ReLU(inplace=True)),

        ]))
        
        if initialize_weights:
            self.initialize_weights()
    
    def forward(self, image : Tensor) -> Tensor:
        """Forward-passes image through feature extractor.
        
        Args:
            input (Tensor): input image.
        
        Returns:
            features (Tensor): features extracted from image.
        
        """
        features = self.extract_features(image)
        return features
    
    def initialize_weights(self) -> None:
        """Initializes weights of feature extractor with ImageNet-1k-trained weights of VGG16.
        """
        print('initializing weights ...')
        vgg16 = models.vgg16(pretrained = True)
        net_keys = list(self.extract_features.state_dict().keys())
        vgg_keys = list(vgg16.state_dict().keys())
        for i in range(len(self.extract_features.state_dict().items())-2):
            self.extract_features.state_dict()[net_keys[i]].data[:] = \
            vgg16.state_dict()[vgg_keys[i]].data[:]
        
        
