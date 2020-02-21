#!/usr/bin/env python
# coding: utf-8

# In[260]:


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


# In[261]:


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


# In[262]:


class MLP(nn.Module):

    def __init__(self, n_classes=10,n1=256,n2=64,n3=16):
        
        super(MLP, self).__init__()
        self.bn1 = nn.BatchNorm1d(784)
        self.bn2 = nn.BatchNorm1d(n1)
        self.bn3 = nn.BatchNorm1d(n2)
        self.fc1 = nn.Linear(784, n1)
        self.fc2 = nn.Linear(n1, n2)
        self.fc3 = nn.Linear(n2, n3)

        self.clf = nn.Linear(n3, n_classes)

    def forward(self, x):
        noise = 0.2
        x = x.view(-1, 784) 
        x = x + noise*torch.normal(torch.zeros(x.shape[0],x.shape[1]), torch.ones(x.shape[0],x.shape[1]))
        x = self.bn1(x)
        x = F.relu(self.bn2(self.fc1(x)))
        x = F.relu(self.bn3(self.fc2(x)))
        x = F.relu(self.fc3(x))
        out = self.clf(x)

        return out


# In[263]:


def train_one_epoch(model, trainloader, optimizer, device):
    model.train()
    avg_loss = AverageMeter("average-loss")
    avg_accr = AverageMeter("average-accr")
    for batch_idx, (img, target) in enumerate(trainloader):
        #print(img.shape,target.shape)
        img = Variable(img).to(device)
        target = Variable(target).to(device)

        #print(img)
        # Zero out the gradients
        optimizer.zero_grad()

        # Forward Propagation
        out = model(img)
        
        loss = F.cross_entropy(out, target)
        
        equality = (target.data == out.max(dim=1)[1])
        accuracy = equality.type(torch.FloatTensor).mean()

        # backward propagation
        loss.backward()
        avg_loss.update(loss, img.shape[0])
        avg_accr.update(accuracy, img.shape[0])
        # Update the model parameters
        optimizer.step()

    return avg_loss.avg, avg_accr.avg


# In[264]:


def validate_one_epoch(model, valloader, device):
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


# In[267]:


if __name__ == "__main__":

    number_epochs = 16
    

    device = torch.device('cpu')  # Replace with torch.device("cuda:0") if you want to train on GPU
    trans_img = transforms.Compose([transforms.ToTensor()])
    dataset = FashionMNIST("./data/", train=True, transform=trans_img, download=True)
    testset = FashionMNIST("./data/", train=False, transform=trans_img, download=True)
    testloader = DataLoader(testset, batch_size=1024, shuffle=True)
    itr = 0
    model = []
    parameter = []
    for n1 in range(9):
        for n2 in range(9):
            for n3 in range(9):
                n1Inp = 8+(n1)*64
                n2Inp = 8+(n2)*32
                n3Inp = 8+(n3)*16
                model.append(MLP(10,n1Inp,n2Inp,n3Inp).to(device))
                
                #print(len(trainloader.dataset),len(testloader.dataset))
                optimizer = optim.Adam(model[itr].parameters(), lr=0.01)

                #track_loss = []
                #track_accr = []
                #track_vacc = []
                loss=0
                accr=0
                val_accr=0
                for i in range(number_epochs): #tqdm(range(number_epochs)):
        
                    trainset, valset = random_split(dataset, [40000, 20000])
                    trainloader = DataLoader(trainset, batch_size=1024, shuffle=True)
                    valloader = DataLoader(valset, batch_size=1024, shuffle=True)
        
                    loss, accr = train_one_epoch(model[itr], trainloader, optimizer, device)
                    val_loss, val_accr = validate_one_epoch(model[itr], valloader, device)
                    #print("loss: ",loss.item()," acc: ",accr.item()," valAcc: ",val_accr.item())
                    #track_loss.append(loss)
                    #track_accr.append(accr)
                    #track_vacc.append(val_accr)
                print("Iteration: ",itr," loss: ",loss.item()," acc: ",accr.item()," val_loss: ",val_loss.item()," valAcc: ",val_accr.item())
                parameter.append(np.array([n1Inp,n2Inp,n3Inp,loss.item(),accr.item(),val_loss.item(),val_accr.item()]))
                itr=itr+1
                #plt.figure()
                #plt.plot(track_loss)
                #plt.figure()
                #plt.plot(track_accr)
                #plt.plot(track_vacc)
                #plt.title("training-loss-MLP")
                #plt.savefig("./img/training_mlp.jpg")

                #torch.save(model.state_dict(), "./models/MLP.pt")
                
    
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
    # test accuracy


# In[ ]:




