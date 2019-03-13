import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

#kernel_size was not specified, so will probably need to play around
import logging
import torch.nn as nn
import random
import numpy as np
import torch
import torch.autograd as autograd

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class DQN(nn.Module):
    """
    Base class for all models
    """
    def __init__(self, num_classes=242):
        super(DQN, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.global_layer1 = nn.Sequential(
        nn.Conv2d(4, 32, kernel_size=50, stride=4),
        nn.BatchNorm2d(32),
        nn.ReLU()
        )
        self.global_layer2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=2, stride=2),
        nn.BatchNorm2d(64),
        nn.ReLU()
        )
        self.global_layer3 = nn.Sequential(
        nn.Conv2d(64, 64, stride=1, kernel_size=2),
        nn.BatchNorm2d(64),
        nn.ReLU()
        )
        self.local_layer = nn.Sequential(
        nn.Conv2d(1, 128, stride=1, kernel_size=1),
        nn.BatchNorm2d(128),
        nn.ReLU()
        )
        self.fc1 = nn.Linear(16064, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x, y):
        """
        Forward pass logic

        :return: Model output
        """
        #Assume x is of for (batch_size, 4 , 81, 81)
        #Assume y is of for (batch_size, 2 , 11, 11)
        x = x.float()
        y = y.float()
        x = self.global_layer1(x) 
        x = self.global_layer2(x) 
        x = self.global_layer3(x) 
        y = self.local_layer(y)
        x = (x.reshape(x.size(0), -1))
        y = (y.reshape(x.size(0), -1))
        x = np.squeeze(x)
        y = np.squeeze(y)
        cat = torch.cat([x,y])
        out = self.fc1(cat)
        out = self.fc2(out)
        return out

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action  = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)
        return action

