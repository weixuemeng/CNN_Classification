import torch

import torch.nn as nn
import torch.nn.functional as F


class FNN(nn.Module):
    def __init__(self, loss_type, num_classes):
        super(FNN, self).__init__()

        self.loss_type = loss_type
        self.num_classes = num_classes

        """add your code here"""
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)


    def forward(self, x):
        output = None

        """add your code here"""
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim =1 )

        return x

    def get_loss(self, output, target):
        loss = None

        """add your code here"""
        criterion = nn.MSELoss()
        decay = 0.0001

        # Calculate the loss
        # loss = criterion(output, target) 
        if self.loss_type== "ce":
            loss = F.cross_entropy(output, target)
        else:
            
            loss = criterion(output, target)  # mse error loss: 

        return loss