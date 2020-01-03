import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


batch_size = 128
ngf = 25
ndf = 25


def sample_z(batch_size ,z_dim, stpye = '0'):
    if stpye == '0':
        return torch.randn(batch_size, z_dim,requires_grad=False)
    elif stpye == '1':
        return torch.randn(batch_size, z_dim ,1, 1,requires_grad=False)


class flatten(nn.Module):
    def forward(self, X):
        return X.view(X.shape[0], -1)

class unFlatten(nn.Module):
    def forward(self, input, size=128*2*2):
        return input.view(input.size(0), -1, 1, 1)

class Discriminator1(nn.Module):
    def __init__(self, Data = 'MNIST'):
        super(Discriminator1, self).__init__()
        
        if Data == 'MNIST':
            self.dim = {'batch_size': batch_size, 'hight':28, 'pixels':784, 'channels':1}
        elif Data == 'CIFAR':
            self.dim = {'batch_size': batch_size, 'hight':32, 'pixels':1024, 'channels':3}
        else: raise ValueError("Data set not support")
              
        if Data == 'MNIST':
          self.model = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(1, ndf, 4, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout2d(p=0.15),
            # state size. (ndf) x 13 x 13
            nn.Conv2d(ndf, ndf * 2, 5, 2, 0, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 5 x 5
            #nn.Dropout2d(p=0.15),
            nn.Conv2d(ndf * 2, 1, 5, 1, 0, bias=False),
            nn.Sigmoid(),
            # state size. (1) x 1 x 1
            flatten()
        )
        elif Data == 'CIFAR':
          self.model = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(3, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 6, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 6),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 6, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            flatten()
        )


    
    def flatten(self, X):     
        return X.view(X.shape[0], -1)   #batch_size* (pixels*channels)
    
    def forward(self, X):    
        """
        X:batch_szie*channel*hight*width
        ---
        out: batch_szie*(channel*hight*width)
        """
        #X = self.flatten(X) 

        return self.model(X)

class Generator1(nn.Module):
    def __init__(self, z_dim,Data = 'MNIST'):
        super(Generator1, self).__init__()
        
        if Data == 'MNIST':
            self.dim = {'batch_size': batch_size, 'hight':28, 'pixels':784, 'channels':1}
        elif Data == 'CIFAR':
            self.dim = {'batch_size': batch_size, 'hight':32, 'pixels':1024, 'channels':3}
        else: raise ValueError("Data set not support")
        
        self.z_dim = z_dim
        
        if Data == 'MNIST':
    
            """
            self.model = nn.Sequential(
            unFlatten(),
            # input is Z, going into a convolution zdim,1,1
            nn.ConvTranspose2d( z_dim, ngf * 4, 5, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(),
            # state size. (ngf*8) x 5 x 5
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 5, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(),
            # state size. (ngf*4) x 13 x 13
            nn.ConvTranspose2d(ngf * 2, 1, 4, 2, 0, bias=False),
            nn.Tanh()
            # state size. (ngf*4) x 28 x 28
            

            # state size. (1) x 28 x 28
            )

            """
            self.model = nn.Sequential(
            unFlatten(),
            # input is Z, going into a convolution zdim,1,1
            nn.ConvTranspose2d( z_dim, ngf * 8, 5, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 6),
            nn.ReLU(),
#            nn.Dropout2d(p=0.5),
            # state size. (ngf*8) x 5 x 5
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 5, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(),
#            nn.Dropout2d(p=0.5),
            # state size. (ngf*4) x 9 x 9
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 5, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(),
#            nn.Dropout2d(p=0.5),
            # state size. (ngf*4) x 21 x 21
            nn.ConvTranspose2d(ngf * 2, ngf * 1, 5, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 1),
            nn.ReLU(),
#            nn.Dropout2d(p=0.5),
            # state size. (ngf*4) x 25 x 25
            nn.ConvTranspose2d( ngf * 1, 1, 4, 1, 0, bias=False),
            nn.Tanh()
            # state size. (1) x 28 x 28
            )
            
   
        elif Data == 'CIFAR':
            self.model = nn.Sequential(
            unFlatten(),
            # input is Z, going into a convolution: B,z_dim,1,1
            nn.ConvTranspose2d(z_dim, ngf * 6, 5, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 6),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            # state size. (ngf*8) x 5 x 5
            nn.ConvTranspose2d(ngf * 6, ngf * 6, 5, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 6),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            # state size. (ngf*4) x 9 x 9
            nn.ConvTranspose2d( ngf * 6, ngf * 4, 5, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            # state size. (ngf*2) x 21 x 21
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 5, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
            # state size. (ngf) x 25 x 25
            nn.ConvTranspose2d( ngf * 2, ngf * 1, 5, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),    
            # state size. (ngf) x 29 x 29
            #nn.ConvTranspose2d( ngf * 2, ngf * 1, 5, 1, 0, bias=False),
            #nn.BatchNorm2d(ngf * 1),
            #nn.ReLU(),
            #nn.Dropout2d(p=0.5),
            nn.ConvTranspose2d( ngf * 1, 3, 4, 1, 0, bias=False),
            nn.Tanh()
            )
   

    
    def forward(self, X):    
        """ 
        X:batch_size*z_dim/ batch_size*z_dim*1*1
        ---
        out: batch_szie*(channel*hight*width)
        """
        return self.model(X)



