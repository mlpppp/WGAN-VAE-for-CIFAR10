# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:10:21 2019

@author: funrr
"""
import os
from torch.utils.data import DataLoader
import numpy as np
import torchvision 
import matplotlib.pyplot as plt

os.chdir(r"C:\Users\funrr\Desktop\DL3")
work_dir = os.getcwd()
p = os.path.join(work_dir, 'DATA')
batch_size = 300

def MNISTLoader(path, download = False, batch_size=batch_size):
    #shape: batch_size*1*28*28
    train_loader = DataLoader(\
        torchvision.datasets.MNIST(path,download= download,\
            train=True, \
            transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),\
            torchvision.transforms.Normalize((0.1,),(0.5,))])),\
        batch_size = batch_size, shuffle=True)

    test_loader = DataLoader(\
        torchvision.datasets.MNIST(path,download= download,\
            train=False, \
            transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),\
            torchvision.transforms.Normalize((0.1,),(0.5,))])),\
        batch_size = batch_size, shuffle=True)

    return train_loader, test_loader


def CIFAR10Loader(path, download = False, batch_size=batch_size):
    #shape: batch_size*3*32*32
    
    train_loader = DataLoader(\
        torchvision.datasets.CIFAR10(path,download= download,\
            train=True, \
            transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),\
            torchvision.transforms.Normalize((0.1,),(0.5,))])),\
        batch_size = batch_size, shuffle=True)

    test_loader = DataLoader(\
        torchvision.datasets.CIFAR10(path,download= download,\
            train=False, \
            transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),\
            torchvision.transforms.Normalize((0.1,),(0.5,))])),\
        batch_size = batch_size, shuffle=True)

    return train_loader, test_loader

def GetDataDir():
    return p
