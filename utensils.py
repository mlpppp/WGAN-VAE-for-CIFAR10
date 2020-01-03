# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 18:02:20 2019

@author: funrr
"""
import os
import numpy as np
import DATA
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image

batch_size = 300
os.chdir(r"C:\Users\funrr\Desktop\DL3")
work_dir = os.getcwd()
p = os.path.join(work_dir, 'DATA')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plotCIFAR(img):   #img:3*32*32
    plt.imshow(np.moveaxis(img.detach().numpy(), 0, 2))
    
def plotMNIST(img): #img:1*28*28
    plt.imshow(img.view(28,28))
    
def flatten(X):
    """
    X:batch_size*channels*hight*width
    ---
    batch_size* (pixels*channels)
    
    """
    return X.view(X.shape[0], -1)   #batch_size* (pixels*channels)


def deflatten(X, Data = 'MNIST'):  
    """
    X:batch_size* (pixels*channels) 
    Data:'MNIST' or 'CIFAR'
    ---
    batch_size*channels*hight*width
    """
    if Data == 'MNIST':
        return X.view(X.shape[0], 1, 28, 28)
    elif Data == 'CIFAR':
        return X.view(X.shape[0], 3, 32, 32)
    else: raise ValueError("Data set not support")
  
    
def test(Data, model, n = 6):
    """
    Data:'MNIST' or 'CIFAR'
    """
    if Data == 'MNIST':
        _, test_loader = DATA.MNISTLoader(p, download = False, batch_size=batch_size)
    elif Data == 'CIFAR':
        _, test_loader = DATA.CIFAR10Loader(p, download = False, batch_size=batch_size)
    else: raise ValueError("Data set not support")
    
    model = model.eval()
    with torch.no_grad():
        idx, (X,_) = next(enumerate(test_loader))
        X = X.to(device)
        X_bar, mu, logvar = model.forward(X)
        if  Data == 'MNIST':             
            img = torch.cat([X[:n], X_bar.view(batch_size, 1, 28, 28)[:n]])    
            plt.pause(0.001)
            plt.imshow(img.view(28*n*2,28*1).cpu())
            #save_image(img.cpu(),
                         #'C:/Users/funrr/Desktop/DL3/results/test0' + '.png', nrow=n)
        elif Data == 'CIFAR':
            img = torch.cat([X[:n], X_bar.view(batch_size, 3, 32, 32)[:n]])  
            plt.pause(0.001)
            plt.imshow(np.moveaxis(img.cpu().numpy(), 1, 3).reshape(n*2*32,32,3))
            #save_image(img.cpu(),
                        # 'C:/Users/funrr/Desktop/DL3/results/test1' + '.png', nrow=n)
            
        
        
    
    
