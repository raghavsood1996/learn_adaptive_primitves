#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import csv
import sys
import numpy as np
import base64
import os
import random
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg


# In[21]:


def getdataloaders(data,batch_size=50):
    np.random.shuffle(data.T)
    train_data1 = data[:,:75000]
    test_data1 = data[:,75000:]
    
    #creating train batches
    num_batches_train = int(train_data1.shape[1]/batch_size)
    train_data= np.empty([num_batches_train,train_data1.shape[0],batch_size])
    for idx in range (0, num_batches_train):
        train_data[idx,:,:]=train_data1[:,idx*batch_size:(idx + 1)*batch_size]
    
    #creating test batches
    num_batches_test = int(test_data1.shape[1]/batch_size)
    test_data =np.empty([num_batches_test,test_data1.shape[0],batch_size])
    for idx in range (0, num_batches_test):
        test_data[idx,:,:]=test_data1[:,idx*batch_size:(idx + 1)*batch_size]
    
    return train_data,test_data


# In[97]:


class network(nn.Module):
    def __init__(self,input_nodes,hidden_nodes1,hidden_nodes2,output_node):
        super(network,self).__init__()
        self.fc1 = nn.Linear(input_nodes,hidden_nodes1)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_nodes1)
        
        self.fc2 = nn.Linear(hidden_nodes1,hidden_nodes2)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.bn2 = nn.BatchNorm1d(num_features = hidden_nodes2)
        
        self.fc3 = nn.Linear(hidden_nodes2,output_nodes)
        nn.init.xavier_uniform_(self.fc3.weight)
        
        self.out_act = nn.Sigmoid();
        
    def forward(self,X):
        X = F.relu(self.bn1(self.fc1(X)))
        X = self.fc2(X)
        X = F.dropout2d(X,p=0.5)
        X = F.relu(X)
        X = self.fc3(X)
        out = self.out_act(X)
        return out
        
    


# In[98]:


def train(trainloader, optimizer, criterion, epoch, net):
    net.train()
    train_loss_sum = 0
    for  idx in range(trainloader.size(0)):
        target = trainloader[idx,362,:].reshape(trainloader.size(2)).float().reshape(trainloader.size(2),1)
        input_vectors = trainloader[idx,:362,:].float().t()
        output = net.forward(input_vectors)
        loss = criterion(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_sum += float(loss)
    train_loss = train_loss_sum/trainloader.size(0)
    print("Epoch No." , epoch, " Train Loss" ,train_loss)
    return train_loss


def test(testloader,criterion,epoch,net):
    net.eval()
    test_loss_sum = 0
    for idx in range(testloader.size(0)):
        target = testloader[idx,362,:].reshape(testloader.size(2)).float().reshape(testloader.size(2),1)
        input_vectors = testloader[idx,:362,:].float().t()
        output = net.forward(input_vectors)
        loss = criterion(output,target.float())
        test_loss_sum += float(loss)
    test_loss = test_loss_sum/testloader.size(0)
    print("Epoch No." , epoch, " Test Loss" ,test_loss)
    print(" ")
    return test_loss
        
        
    


# In[99]:


def plot_loss(x,y):
    plt.plot(x,y)
    plt.show()


# In[100]:


input_nodes = 362
hidden_nodes1 = 150
hidden_nodes2 = 60
output_nodes = 1
batch_number = 50
learning_rate = 0.01
epochs = 100

data=np.load("../data/data_float.npy")
train_data,test_data = getdataloaders(data,10)
trainloader = torch.from_numpy(train_data)
testloader =  torch.from_numpy(test_data)


reeds_net = network(input_nodes,hidden_nodes1,hidden_nodes2,output_nodes)
optimizer = torch.optim.Adam(reeds_net.parameters(), lr=learning_rate)
criterion = nn.BCELoss()

train_loss_list = []
test_loss_list = []
epoch_list = []
for epoch in range(epochs):
    train_loss=train(trainloader, optimizer, criterion, epoch, reeds_net)
    test_loss= test(testloader, criterion, epoch, reeds_net)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    epoch_list.append(epoch)


# In[ ]:


#plotting the losses
plt.plot(epoch_list,train_loss_list)
plt.plot(epoch_list,test_loss_list)


# In[ ]:




