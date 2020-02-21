#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pdb
import os
from tqdm import tqdm

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
from torchvision.datasets import FashionMNIST
from torchvision import transforms


# In[3]:


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


# In[4]:


class LeNet(nn.Module):

    def __init__(self, n_classes=10,n1=32,n2=64,emb_dim=20):
        
        '''
        Define the initialization function of LeNet, this function defines
        the basic structure of the neural network
        '''

        super(LeNet, self).__init__()
        self.bn1 = nn.BatchNorm2d(1) 
        self.conv1 = nn.Conv2d(1, n1, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=5, stride=1, padding=2)
        self.emb = nn.Linear(n2*7*7, emb_dim)
        self.clf = nn.Linear(emb_dim, n_classes)

    def num_flat_features(self, x):
        '''
        Calculate the total tensor x feature amount
        '''

        size = x.size()[1:]  # All dimensions except batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features

    def forward(self, x):
        noise = 0.2
        x = x.view(-1, 1, 28, 28)
        x = x + noise*torch.normal(torch.zeros(x.shape[0],x.shape[1],x.shape[2],x.shape[3]), torch.ones(x.shape[0],x.shape[1],x.shape[2],x.shape[3]))
        x = self.bn1(x)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.emb(x)
        out = self.clf(x)

        return out


# In[5]:


def train_one_epoch(model, trainloader, optimizer, device):
    """ Training the model using the given dataloader for 1 epoch.

    Input: Model, Dataset, optimizer, 
    """

    model.train()
    avg_loss = AverageMeter("average-loss")
    avg_accr = AverageMeter("average-accr")
    for batch_idx, (img, target) in enumerate(trainloader):
        img = Variable(img).to(device)
        target = Variable(target).to(device)

        # Zero out the gradients
        optimizer.zero_grad()

        # Forward Propagation
        prob = model(img)
        loss = F.cross_entropy(prob, target)
        equality = (target.data == prob.max(dim=1)[1])
        accuracy = equality.type(torch.FloatTensor).mean()

        # backward propagation
        loss.backward()
        avg_loss.update(loss, img.shape[0])
        avg_accr.update(accuracy, img.shape[0])
        # Update the model parameters
        optimizer.step()

    return avg_loss.avg, avg_accr.avg


# In[6]:


def validate_one_epoch(model, valloader, device):
    #print("48543876566")
    avg_loss = AverageMeter("val_average-loss")
    avg_accr = AverageMeter("val_average-accr")
    for batch_idx, (img, target) in enumerate(valloader):
        img = Variable(img).to(device)
        target = Variable(target).to(device)
        out = model(img)
        loss = F.cross_entropy(out, target)
        equality = (target.data == out.max(dim=1)[1])
        accuracy = equality.type(torch.FloatTensor).mean()
        avg_loss.update(loss, img.shape[0])
        avg_accr.update(accuracy, img.shape[0])
    
    return avg_loss.avg, avg_accr.avg


# In[ ]:


if __name__ == "__main__":

    number_epochs = 6

    # Use torch.device("cuda:0") if you want to train on GPU
    # OR Use torch.device("cpu") if you want to train on CPU
    device = torch.device('cpu')

    model = LeNet(10).to(device)

    trans_img = transforms.Compose([transforms.ToTensor()])
    dataset = FashionMNIST("./data/", train=True, transform=trans_img, download=True)
    testset = FashionMNIST("./data/", train=False, transform=trans_img, download=True)
    testloader = DataLoader(testset, batch_size=1024, shuffle=True)

    

    #track_loss = []
    #track_accr = []
    #track_vacc = []
    
    itr = 0
    model = []
    parameter = []
    
    for n1 in range(9):
        for n2 in range(9):
            for n3 in range(5):
                n1Inp = 8+(n1)*32
                n2Inp = 8+(n2)*32
                n3Inp = 8+(n3)*2
                
                model.append(LeNet(10,n1Inp,n2Inp,n3Inp).to(device))
                optimizer = optim.Adam(model[itr].parameters(), lr=0.01)
                loss=0
                accr=0
                val_accr=0
                for i in range(number_epochs):
                    trainset, valset = random_split(dataset, [48000, 12000])
                    trainloader = DataLoader(trainset, batch_size=1024, shuffle=True)
                    valloader = DataLoader(valset, batch_size=1024, shuffle=True)
                    loss, accr = train_one_epoch(model[itr], trainloader, optimizer, device)
                    val_loss, val_accr = validate_one_epoch(model[itr], valloader, device)
                    
                    
                print("Iteration: ",itr," loss: ",loss.item()," acc: ",accr.item()," val_loss: ",val_loss.item()," valAcc: ",val_accr.item())
                parameter.append(np.array([n1Inp,n2Inp,n3Inp,loss.item(),accr.item(),val_loss.item(),val_accr.item()]))
                itr=itr+1        #track_loss.append(loss)
                #track_accr.append(accr)
                #track_vacc.append(val_accr)
                #plt.figure()
                #plt.plot(track_loss)
                #plt.title("training-loss-ConvNet")
                #plt.savefig("./img/training_convnet.jpg")

                #torch.save(model.state_dict(), "./models/convNet.pt")
                
    parameter = np.array(parameter)
    plt.figure()
    plt.scatter(parameter[:,0]+parameter[:,1]+parameter[:,2],parameter[:,4])
    plt.scatter(parameter[:,0]+parameter[:,1]+parameter[:,2],parameter[:,6])
    plt.title("TestAcc vs ValAcc")
    plt.xlabel('Model-Size')
    plt.ylabel('Accuracy')
    
    plt.figure()
    plt.scatter(parameter[:,0]+parameter[:,1]+parameter[:,2],parameter[:,3])
    plt.scatter(parameter[:,0]+parameter[:,1]+parameter[:,2],parameter[:,5])
    plt.title("TestLoss vs ValLoss")
    plt.xlabel('Model-Size')
    plt.ylabel('Loss')
    plt.show()


# In[1]:


parameter = np.array(parameter)
plt.figure()
plt.scatter(parameter[:,0]+parameter[:,1]+parameter[:,2],parameter[:,4])
plt.scatter(parameter[:,0]+parameter[:,1]+parameter[:,2],parameter[:,6])
plt.title("TestAcc vs ValAcc")
plt.xlabel('Model-Size')
plt.ylabel('Accuracy')
    
plt.figure()
plt.scatter(parameter[:,0]+parameter[:,1]+parameter[:,2],parameter[:,3])
plt.scatter(parameter[:,0]+parameter[:,1]+parameter[:,2],parameter[:,5])
plt.title("TestLoss vs ValLoss")
plt.xlabel('Model-Size')
plt.ylabel('Loss')
plt.show()


# In[ ]:




