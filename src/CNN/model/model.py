import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

#kernel_size was not specified, so will probably need to play around
import logging
import torch.nn as nn
import numpy as np


class CNN(nn.Module):
    """
    Base class for all models
    """
    def __init__(self, num_classes=242):
        super(cnn, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.global_layer1 = nn.Sequential(
        nn.Conv2d(4, 32, kernel_size=2, stride=4),
        nn.BatchNorm2d(32),
        nn.ReLU()
        )
        self.global_layer2 = nn.Sequential(
        nn.Conv2d(32, 64, stride=2, kernel_size=2)
        nn.BatchNorm2d(64),
        nn.ReLU()
        )
        self.global_layer3 = nn.Sequential(
        nn.Conv2d(64, 64, stride=1, kernel_size=2)
        nn.BatchNorm2d(64),
        nn.ReLU()
        )
        self.local_layer = nn.Sequential(
        self.local_conv = nn.Conv2d(2, 128, stride=1, kernel_size=2)
        nn.BatchNorm2d(128),
        nn.ReLU()
        )
        self.fc1 = nn.Linear(192, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x, y):
	"""
	Forward pass logic

	:return: Model output
	"""
        #Assume x is of for (batch_size, 4 , 81, 81)
        #Assume y is of for (batch_size, 2 , 11, 11)
        x = self.global_layer1(x) 
        x = self.global_layer2(x) 
        x = self.global_layer3(x) 
        y = self.local_layer(y)
        cat = np.concat([x,y])
        out = self.fc1(cat)
        out = self.fc2(cat)
        return out
        
    def summary(self):
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + '\nTrainable parameters: {}'.format(params)
        # print(super(BaseModel, self))




