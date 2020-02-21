import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Load the fashion-mnist pre-shuffled train data and test data
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

class MLP(nn.Module):

    #def __init__(self, n_classes=10,n1=320,n2=96,n3=32): #Dense | Test Accuracy:  0.8992999792098999  Test Loss:  0.29948028922080994
    #def __init__(self, n_classes=10,n1=384,n2=128,n3=32): #Dense | Test Accuracy:  0.9039000272750854  Test Loss:  0.2879541516304016
    def __init__(self, n_classes=10,n1=384,n2=128,n3=64): #Dense | Test Accuracy:  0.8995000123977661  Test Loss:  0.300533652305603
        
        super(MLP, self).__init__()
        self.bn1 = nn.BatchNorm1d(784)
        self.bn2 = nn.BatchNorm1d(n1)
        self.bn3 = nn.BatchNorm1d(n2)
        self.fc1 = nn.Linear(784, n1)
        self.fc2 = nn.Linear(n1, n2)
        self.fc3 = nn.Linear(n2, n3)

        self.clf = nn.Linear(n3, n_classes)

    def forward(self, x):
        
        x = x.view(-1, 784) 
        #x = x + 
        #x = self.bn1(x)
        x = F.relu(self.bn2(self.fc1(x)))
        x = F.relu(self.bn3(self.fc2(x)))
        x = F.relu(self.fc3(x))
        out = self.clf(x)

        return out

def test_one_epoch(model, valloader, device):
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

if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    xTrainCNN = x_train.reshape((60000, 28, 28, 1))
    xTestCNN = x_test.reshape((10000, 28, 28, 1))
    xTrainCNN = xTrainCNN
    xTestCNN = xTestCNN

    yTest = np.zeros((y_test.shape[0],10))
    yTrain = np.zeros((y_train.shape[0],10))
    i=0
    for ele in y_test:
        yTest[i][ele] = 1
        i=i+1
    i=0
    for ele in y_train:
        yTrain[i][ele] = 1
        i=i+1

    cmodel = tf.keras.models.load_model('FMCNN.h5')

    a = (cmodel.predict(xTestCNN))
    oitr = np.argmax(a,axis=1)

    metric = cmodel.evaluate(xTestCNN,yTest,32)
    o2 = open('convolution-neural-net.txt','a')
    o2.write("Loss on Test Data : "+str(metric[0])+"\n")
    o2.write("Accuracy on Test Data : "+str(metric[1])+"\n")
    o2.write("gt_label,pred_label"+"\n")
    for i in range(10000):
        o2.write(str(y_test[i])+','+str(oitr[i])+'\n')

    o2.close()
    #--------------------------------------------------------------------------------------------
    device = torch.device('cpu')  # Replace with torch.device("cuda:0") if you want to train on GPU
    trans_img = transforms.Compose([transforms.ToTensor()])
    dataset = FashionMNIST("./data/", train=True, transform=trans_img, download=True)
    testset = FashionMNIST("./data/", train=False, transform=trans_img, download=True)
    testloader = DataLoader(testset, batch_size=1024, shuffle=True)
    number_epochs=1
    # dense
    dmodel = torch.load("denseModel.pt")
    o1 = open('multi-layer-net.txt','a')
    loss, accr = test_one_epoch(dmodel, testloader, device)
    o1.write("Loss on Test Data : "+str(loss.item())+"\n")
    o1.write("Accuracy on Test Data : "+str(accr.item())+"\n")
    o1.write("gt_label,pred_label"+"\n")
    iloop=0
    for batch_idx, (img, target) in enumerate(testloader):
        img = Variable(img).to(device)
        target = Variable(target).to(device)
        out = dmodel(img)
        for itr in range(img.shape[0]):
            o1.write(str(target.data[itr].item())+","+str(out.max(dim=1)[1][itr].item())+"\n")

    print("Dense | Test Accuracy: ",accr.item()," Test Loss: ",loss.item())
    o1.close()
