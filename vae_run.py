# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 16:37:39 2019

@author: funrr
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import DATA
import VAE
import utensils
import matplotlib.pyplot as plt

os.chdir(r"C:\Users\funrr\Desktop\DL3")
work_dir = os.getcwd()
p = os.path.join(work_dir, 'DATA')

batch_size = 300

train_loader,_ = DATA.MNISTLoader(p, download = False, batch_size=batch_size)
#idx, (X,_) = next(enumerate(train_loader))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vae = VAE.MLP_VAE(Data = 'MNIST',batch_size= batch_size, z_dim=30).to(device)


optimizer = optim.Adam(vae.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
loss_log = []
for epoch in range(20):  
        for step, (train_data, _) in enumerate(train_loader):         
            optimizer.zero_grad() 
            
            x_bar, mu, logvar = vae.forward(train_data.to(device) )        #batch*1*28*28    
            loss = VAE.loss_function(x_bar, train_data.view(-1, 784).to(device), mu, logvar)    
            print('irer'+ str(step), "loss:" + str(loss.item()))
            loss_log.append(loss)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            if step % 6 == 0:
                print('epoch:' + str(epoch), 'irer'+ str(step), "loss:" + str(loss.item()))
                
            if step % 60 == 0:
                utensils.test('MNIST', vae, n = 2)
                fig1 = plt.figure()
                ax1 = fig1.add_subplot(111)
                ax1.plot(loss_log, label='loss')
                ax1.legend(prop={'size': 9})

                title = "VAE loss"
                ax1.set_title(title)
                ax1.set_xlabel("*#batchs")
                ax1.set_ylabel("loss")
                plt.pause(0.001)
                fig1
                

plt.plot(loss_log)



#-------------------------------------

train_loader,_ = DATA.CIFAR10Loader(p, download = False, batch_size=batch_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpxvae = VAE.Cpx_VAE(Data = 'CIFAR',batch_size= 300, z_dim=128).to(device)
optimizer = optim.Adam(cpxvae.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
loss_log = []
for epoch in range(20):  
        for step, (train_data, _) in enumerate(train_loader):  
   
            optimizer.zero_grad() 
            
            x_bar, mu, logvar = cpxvae.forward(train_data.to(device) )            #batch*1*28*28    
            loss = VAE.loss_function(x_bar, train_data.view(train_data.size(0), -1).to(device), mu, logvar)   
            if step % 6 == 0:
                print('epoch:' + str(epoch), 'irer'+ str(step), "loss:" + str(loss.item()))
                
            if step % 60 == 0:
                utensils.test('CIFAR', cpxvae, n = 2)
                fig1 = plt.figure()
                ax1 = fig1.add_subplot(111)
                ax1.plot(loss_log, label='loss')
                ax1.legend(prop={'size': 9})

                title = "VAE loss"
                ax1.set_title(title)
                ax1.set_xlabel("*#batchs")
                ax1.set_ylabel("loss")
                plt.pause(0.001)
                fig1
                
            loss_log.append(loss)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
plt.plot(loss_log)



#DATA.plotMNIST(Xr[0].cpu().detach())

#----------------------------------------------------
train_loader,_ = DATA.CIFAR10Loader(p, download = False, batch_size=batch_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cnnvae = VAE.CNN_VAE(Data = 'CIFAR',batch_size= 300, z_dim=128).to(device)
optimizer = optim.Adam(cnnvae.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
loss_log = []
for epoch in range(20):  
        for step, (train_data, _) in enumerate(train_loader):  
   
            optimizer.zero_grad() 
            
            x_bar, mu, logvar = cnnvae.forward(train_data.to(device) )            #batch*1*28*28    
            loss = VAE.loss_function(x_bar.view(x_bar.size(0), 3072), train_data.view(train_data.size(0), 3072).to(device), mu, logvar)   
            
            if step % 6 == 0:
                print('epoch:' + str(epoch), 'irer'+ str(step), "loss:" + str(loss.item()))
                
            if step % 60 == 0:
                utensils.test('CIFAR', cnnvae, n = 2)
                fig1 = plt.figure()
                ax1 = fig1.add_subplot(111)
                ax1.plot(loss_log, label='loss')
                ax1.legend(prop={'size': 9})
                
                title = "VAE loss"
                ax1.set_title(title)
                ax1.set_xlabel("*#batchs")
                ax1.set_ylabel("loss")

                plt.pause(0.001)
                fig1
                
            loss_log.append(loss)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            plt.plot(loss_log)
            fig1.savefig(r'C:\Users\funrr\Desktop\DL3\results'+ '/temp.jpg',dpi = 600)
