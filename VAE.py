# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:05:58 2019

@author: funrr
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


batch_size = 300



def loss_function(x_bar, x, mu, logvar):
    """
    x_bar:generated x, batch_szie*(channel*hight*width)
    mu: batch_szie*z_dim
    logvar: batch_szie*z_dim
    x: batch_szie*(channel*hight*width)
    """
    BCE = F.binary_cross_entropy(x_bar, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


class MLP_VAE(nn.Module):
    def __init__(self, Data = 'MNIST', batch_size = 100, z_dim=20): #Data = 'MNIST' or 'CIFAR' 
        super(MLP_VAE, self).__init__()      
        
        if Data == 'MNIST':
            self.dim = {'batch_size': batch_size, 'hight':28, 'pixels':784, 'channels':1}
        elif Data == 'CIFAR':
            self.dim = {'batch_size': batch_size, 'hight':32, 'pixels':1024, 'channels':3}
        else: raise ValueError("Data set not support")
        
        self.z_dim = z_dim
        
        if Data == 'MNIST':
            self.encoder = nn.Sequential(
                  nn.Linear(784, 400),
                  nn.ReLU()
            )
        elif Data == 'CIFAR':
            self.encoder = nn.Sequential(
                  nn.Linear(3072, 1024),
                  nn.ReLU(),
                  nn.Linear(1024, 400),
                  nn.ReLU()
            )
            
        self.fc_mu = nn.Linear(400, self.z_dim)
        self.fc_logvar = nn.Linear(400, self.z_dim)

        if Data == 'MNIST':
            self.decoder = nn.Sequential(
                  nn.Linear(self.z_dim, 400),
                  nn.ReLU(),
                  nn.Linear(400, 784),
                  nn.Sigmoid()
                  
            )
        elif Data == 'CIFAR':
            self.decoder = nn.Sequential(
                  nn.Linear(self.z_dim, 400),
                  nn.ReLU(),
                  nn.Linear(400, 1024),
                  nn.ReLU(),
                  nn.Linear(1024, 3072),              
                  nn.Sigmoid()
            )        

    
    def flatten(self, X):     
        return X.view(X.shape[0], -1)   #batch_size* (pixels*channels)
        

    
    def forward(self, X):    
        """
        X:batch_szie*channel*hight*width
        ---
        X_bar: batch_szie*(channel*hight*width)
        mu: batch_szie*z_dim
        logvar: batch_szie*z_dim
        """
        X = self.flatten(X) 
        h = self.encoder(X)
        z, mu, logvar = self.bottleneck(h)
        return self.decoder(z), mu, logvar
            
    def bottleneck(self, h):
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def reparameterize(self, mu, logvar):  #done
        std = torch.exp(0.5*logvar)
        u = torch.randn_like(std)
        return mu + u*std

#---------------------------------------------------------------------
class Cpx_VAE(nn.Module):
    def __init__(self, Data = 'MNIST', batch_size = 100, z_dim=128): #Data = 'MNIST' or 'CIFAR' 
        super(Cpx_VAE, self).__init__()      
        
        if Data == 'MNIST':
            self.dim = {'batch_size': batch_size, 'hight':28, 'pixels':784, 'channels':1}
        elif Data == 'CIFAR':
            self.dim = {'batch_size': batch_size, 'hight':32, 'pixels':1024, 'channels':3}
        else: raise ValueError("Data set not support")
        
        self.z_dim = z_dim
        
        if Data == 'MNIST':
            self.encoder = nn.Sequential(
                  nn.Linear(784, 1024),
                  nn.ReLU(),
                  nn.Linear(1024, 2048),
                  nn.ReLU(),
                  nn.Linear(2048, 1024),
                  nn.ReLU(),
                  nn.Linear(1024, 512),
                  nn.ReLU(),
                  nn.Linear(512, 400),
                  nn.ReLU(),
            )
        elif Data == 'CIFAR':
            self.encoder = nn.Sequential(
                  nn.Linear(3072, 2048),
                  nn.BatchNorm1d(2048),
                  nn.ReLU(),               
                  nn.Linear(2048, 1024),
                  nn.BatchNorm1d(1024),
                  nn.ReLU(),
                  nn.Linear(1024, 512),
                  nn.BatchNorm1d(512),
                  nn.ReLU(),
                  nn.Linear(512, 400),
                  nn.ReLU()
            )
            
        self.fc_mu = nn.Linear(400, self.z_dim)
        self.fc_logvar = nn.Linear(400, self.z_dim)

        if Data == 'MNIST':
            self.decoder = nn.Sequential(
                  nn.Linear(self.z_dim, 400),
                  nn.ReLU(),
                  nn.Linear(400, 1024),
                  nn.ReLU(),
                  nn.Linear(1024, 2048),
                  nn.ReLU(),
                  nn.Linear(2048, 1024),
                  nn.ReLU(),
                  nn.Linear(1024, 784),
                  nn.Sigmoid()
                  
            )
        elif Data == 'CIFAR':
            self.decoder = nn.Sequential(
                  nn.Linear(self.z_dim, 400),
                  nn.ReLU(),
                  nn.Linear(400, 512),
                  nn.ReLU(),
                  nn.Linear(512, 1024),
                  nn.ReLU(),
                  nn.Linear(1024, 2048),
                  nn.ReLU(),
                  nn.Linear(2048, 3072),              
                  nn.Sigmoid()
            )        

    
    def flatten(self, X):     
        return X.view(X.shape[0], -1)   #batch_size* (pixels*channels)
        

    
    def forward(self, X):    
        """
        X:batch_szie*channel*hight*width
        ---
        X_bar: batch_szie*(channel*hight*width)
        mu: batch_szie*z_dim
        logvar: batch_szie*z_dim
        """
        X = self.flatten(X) 
        h = self.encoder(X)
        z, mu, logvar = self.bottleneck(h)
        return self.decoder(z), mu, logvar
            
    def bottleneck(self, h):
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def reparameterize(self, mu, logvar):  #done
        std = torch.exp(0.5*logvar)
        u = torch.randn_like(std)
        return mu + u*std

#---------------------------------------------------------------------
class flatten(nn.Module):
    def forward(self, X):
        return X.view(X.shape[0], -1)

class unFlatten(nn.Module):
    def forward(self, input, size=128*2*2):
        return input.view(input.size(0), -1, 1, 1)
ndf = 64 
ngf = 64       
class CNN_VAE(nn.Module):
    def __init__(self, Data = 'MNIST', batch_size = 100, z_dim=128): #Data = 'MNIST' or 'CIFAR' 
        super(CNN_VAE, self).__init__()      
        
        if Data == 'MNIST':
            self.dim = {'batch_size': batch_size, 'hight':28, 'pixels':784, 'channels':1}
        elif Data == 'CIFAR':
            self.dim = {'batch_size': batch_size, 'hight':32, 'pixels':1024, 'channels':3}
        else: raise ValueError("Data set not support")
        
        self.z_dim = z_dim
        
        if Data == 'MNIST':
            self.encoder = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            flatten(),
            nn.Linear(4096,2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048,1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024,400),            
            )

        elif Data == 'CIFAR':

            self.encoder = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(3, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            
            flatten(),
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512,400),        
        )
            
        self.fc_mu = nn.Linear(400, self.z_dim)
        self.fc_logvar = nn.Linear(400, self.z_dim)

        if Data == 'MNIST':
            self.decoder = nn.Sequential(
            unFlatten(),
            # input is Z, going into a convolution
            nn.ConvTranspose2d( z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()

            )
        elif Data == 'CIFAR':
            self.decoder = nn.Sequential(
            unFlatten(),
            # input is Z, going into a convolution
            nn.ConvTranspose2d( z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()

            )

    
    def flatten(self, X):     
        return X.view(X.shape[0], -1)   #batch_size* (pixels*channels)
        

    
    def forward(self, X):    
        """
        X:batch_szie*channel*hight*width
        ---
        X_bar: batch_szie*(channel*hight*width)
        mu: batch_szie*z_dim
        logvar: batch_szie*z_dim
        """
        h = self.encoder(X)
        z, mu, logvar = self.bottleneck(h)
        return self.decoder(z), mu, logvar
            
    def bottleneck(self, h):
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def reparameterize(self, mu, logvar):  #done
        std = torch.exp(0.5*logvar)
        u = torch.randn_like(std)
        return mu + u*std

