import timeit
from collections import OrderedDict

import torch
from torchvision import transforms, datasets

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split

# import the necessary packages
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear  # fully connected layre
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten



#TO DO: Complete this with your CNN architecture. Make sure to complete the architecture requirements.
#The init has in_channels because this changes based on the dataset.

class Net(nn.Module):
    def __init__(self, in_channels, dataset_name):
        super(Net, self).__init__()
        self.dataset_name = dataset_name
        if dataset_name == "MNIST":
          self.conv1 = nn.Conv2d(in_channels, 6, kernel_size = 5)  #  MNIST : input 1*28*28 [6]* 1*5*5 => [60]*24*24 pooling ==> [60] * 12*12
          self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
          self.fc1 = nn.Linear(256, 120)
          self.fc2 = nn.Linear(120, 10)
        else:
          self.conv1 = nn.Conv2d(in_channels, 32, kernel_size = 3)
          self.conv2 = nn.Conv2d(32, 64, kernel_size= 3)
          self.conv3 = nn.Conv2d(64, 256, kernel_size= 3)
          self.fc1 = nn.Linear(1024, 256) # after conv 2:
          self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)


        #print(x.shape)  #"cifar": torch.Size([1000, 160, 5, 5])
        if self.dataset_name == "MNIST":  # 2 conv layer
          x = x.view(x.size(0), -1)
          x = F.relu(self.fc1(x))
          x = self.fc2(x)
          return F.softmax(x,dim=1)
        else:
          x = F.relu(self.conv3(x))
          x = F.max_pool2d(x, kernel_size=2, stride=2)
          # x = F.relu(self.conv4(x))
          # x = F.max_pool2d(x, kernel_size=2, stride=2)


          x = x.view(x.size(0), -1)
          x = F.relu(self.fc1(x))
          # x = F.relu(self.fc2(x))
          x = self.fc3(x)

          return F.softmax(x, dim = 1)


#Function to get train and validation datasets. Please do not make any changes to this function.
def load_dataset(
        dataset_name: str,
):
    if dataset_name == "MNIST":
        full_dataset = datasets.MNIST('./data', train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))]))

        train_dataset, valid_dataset = random_split(full_dataset, [48000, 12000])

    elif dataset_name == "CIFAR10":
        full_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        train_dataset, valid_dataset = random_split(full_dataset, [38000, 12000])

    else:
        raise Exception("Unsupported dataset.")

    return train_dataset, valid_dataset



#TO DO: Complete this function. This should train the model and return the final trained model.
#Similar to Assignment-1, make sure to print the validation accuracy to see
#how the model is performing.

def train_cnn(epoch,data_loader,model,optimizer, device ):
  for batch_idx, (data, target) in enumerate(data_loader):
    data = data.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(data_loader.dataset),
        100. * batch_idx / len(data_loader), loss.item()))

def eval(data_loader,model,dataset, device):
  loss = 0
  correct = 0
  with torch.no_grad(): # notice the use of no_grad
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        loss += F.cross_entropy(output, target).item()
  loss /= len(data_loader.dataset)
  print(dataset+'set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))


def train(model,train_dataset,valid_dataset,device):
    #Make sure to fill in the batch size.
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size= batch_size_test, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # start training
    eval(valid_loader,model,"Validation", device)
    for epoch in range(1, n_epochs + 1):
        train_cnn(epoch,train_loader,model,optimizer, device)
        eval(valid_loader,model,"Validation", device)

    # Gettign result model
    results = dict(
        model=model
    )

    return results

def CNN(dataset_name, device):

    #CIFAR-10 has 3 channels whereas MNIST has 1.
    if dataset_name == "CIFAR10":
        in_channels= 3
    elif dataset_name == "MNIST":
        in_channels = 1
    else:
        raise AssertionError(f'invalid dataset: {dataset_name}')

    model = Net(in_channels, dataset_name).to(device)
    train_dataset, valid_dataset = load_dataset(dataset_name)

    results = train(model, train_dataset, valid_dataset, device)

    return results

n_epochs = 15
batch_size_train = 200
batch_size_test = 1000
learning_rate = 1e-3
momentum = 0.5
log_interval = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
