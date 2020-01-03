# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:46:37 2019

@author: funrr
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.utils import save_image
import matplotlib.pyplot as plt

def sample_class(real_data, label, classn = 0):
    class_data = real_data[(label == classn).nonzero().flatten()]
    return class_data, class_data.size(0)

os.chdir(r"C:\Users\funrr\Desktop\DL3")
work_dir = os.getcwd()
p = os.path.join(work_dir, 'DATA')
import DATA
import GANs

batch_size = 180
discriminator_steps = 20  ##
z_dim = 150
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
one = torch.FloatTensor([1]).to(device)
generator_cold_time = 1  #wgan10 gan1
detector_cold_time = 1



#-------------gangangangangangangangangangangangangangangangangangangan---------------
#one: DCGAN not Wasserstein
#'CIFAR'

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.utils import save_image
import matplotlib.pyplot as plt

def sample_class(real_data, label, classn = 0):
    class_data = real_data[(label == classn).nonzero().flatten()]
    return class_data, class_data.size(0)

os.chdir(r"C:\Users\funrr\Desktop\DL3")
work_dir = os.getcwd()
p = os.path.join(work_dir, 'DATA')
import DATA
import GANs

batch_size = 800
discriminator_steps = 200  ##
z_dim = 150
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
one = torch.FloatTensor([1]).to(device)
generator_cold_time = 1  #wgan10 gan1
detector_cold_time = 1


generator_cold_time = 8

train_loader,_ = DATA.CIFAR10Loader(p, download = False, batch_size=batch_size)
#idx, (X,_) = next(enumerate(train_loader))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

generator = GANs.Generator1(z_dim = z_dim,Data = 'CIFAR').to(device)
discriminator = GANs.Discriminator1(Data = 'CIFAR').to(device)

gen_optimizer = optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
disc_optimizer = optim.SGD(discriminator.parameters(), lr=0.0005)

CEloss = nn.BCELoss()
JSD_log = []
D_reals = []
D_fakes = []


#fixed_noise1 = torch.randn(1, z_dim, 1, 1, device=device)
#fixed_noise2 = torch.randn(1, z_dim, 1, 1, device=device)
fixed_noise1 = torch.randn(1, z_dim, device=device).detach()
#fixed_noise2 = torch.randn(1, z_dim, device=device).detach()

count = 0

for epoch in range(150):
    for batch_idx, (real_data,label) in enumerate(train_loader):
        real_data, batch_size = sample_class(real_data,label,2)
        disc_optimizer.zero_grad()
        real_data = real_data.to(device)
        fake_data = generator.forward(GANs.sample_z(batch_size ,z_dim).to(device)).detach()   
        
#        noise_real = 0.0000001*torch.randn_like(real_data)          #noise
#        real_data += noise_real
       
        prediction_real = discriminator(real_data)
        loss_true_label = CEloss(prediction_real, (0.1*torch.rand(real_data.size(0),1)+0.9).to(device)) #1=real, 0 =fake
        loss_true_label.backward()
        D_real = prediction_real.mean().item()
        
        prediction_fake = discriminator(fake_data)  
        loss_false_label = CEloss(prediction_fake, (0.1*torch.rand(fake_data.size(0),1)).to(device))
        loss_false_label.backward()
        D_fake = prediction_fake.mean().item()
        disc_optimizer.step()                       
    
                       
        if batch_idx % 6 == 0: 
                print("current JSD*******************")
                print('epoch:'+str(epoch), 'batch:' +str(batch_idx))
                print("JSD:     % .6f" %(JSD.item()))    
                print("D_real:  % .6f " %(D_real))   
                print("D_fake: % .6f " %(D_fake))   
        #if batch_idx > 2: break
        if batch_idx % generator_cold_time == 0:        
            gen_optimizer.zero_grad() 
            gen_fake_data = generator.forward(GANs.sample_z(batch_size ,z_dim).to(device))
            gen_prediction_fake = discriminator(gen_fake_data) 
            loss_FP = CEloss(gen_prediction_fake, torch.ones(gen_fake_data.size(0),1).to(device))
            loss_FP.backward()
            gen_optimizer.step()

            JSD = loss_FP.detach()
            if JSD.item()>=0:
                JSD_log.append(JSD.item())
            
        D_reals.append(D_real)
        D_fakes.append(D_fake)
        torch.cuda.empty_cache()
        if batch_idx % 100 == 0: 
            test_images1 = generator.forward(fixed_noise1.to(device))
            plt.pause(0.001) 
            plt.imshow(np.moveaxis(test_images1.cpu().detach().numpy(), 1, 3).reshape(32,32,3))
            test_images2 = generator.forward(torch.rand(1, z_dim, device=device).detach())
            plt.pause(0.001) 
            plt.imshow(np.moveaxis(test_images2.cpu().detach().numpy(), 1, 3).reshape(32,32,3))
            if  batch_idx % 100 == 0:
                save_image(test_images1.cpu(),
                         r'C:\Users\funrr\Desktop\DL3\results\cif all\a' + str(count)+ '.png')    
                count +=1   
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            ax1.plot(JSD_log, label='JSD')
            ax1.legend(prop={'size': 9})

            title = "JSD curve"
            ax1.set_title(title)
            ax1.set_xlabel("*#batchs")
            ax1.set_ylabel("JSD")
            plt.pause(0.001)
            fig1 = plt.figure()
    
            
            ax1 = fig1.add_subplot(111)
            ax1.plot(D_reals, label='D_real')
            ax1.legend(prop={'size': 9})
   
            title = "D_real curve"
            ax1.set_title(title)
            ax1.set_xlabel("*#batchs")
            ax1.set_ylabel("D_real")
            plt.pause(0.001)
           
            fig1 = plt.figure()
 
            
            ax1 = fig1.add_subplot(111)
            ax1.plot(D_fakes, label='D_fakes')
            ax1.legend(prop={'size': 9})

            title = "D_fakes curve"
            ax1.set_title(title)
            ax1.set_xlabel("*#batchs")
            ax1.set_ylabel("D_fakes")
            plt.pause(0.001)
            
            fig1
            
#-----------wganwganwganwganwganwganwganwganwganwganwganwganwgan--------------
#one: DCGAN Wasserstein
#'CIFAR'
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.utils import save_image
import matplotlib.pyplot as plt

def sample_class(real_data, label, classn = 0):
    class_data = real_data[(label == classn).nonzero().flatten()]
    return class_data, class_data.size(0)

os.chdir(r"C:\Users\funrr\Desktop\DL3")
work_dir = os.getcwd()
p = os.path.join(work_dir, 'DATA')
import DATA
import GANs

batch_size = 800
discriminator_steps = 200  ##
z_dim = 300
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
one = torch.FloatTensor([1]).to(device)
detector_cold_time = 1


generator_cold_time = 15

train_loader,_ = DATA.CIFAR10Loader(p, download = False, batch_size=batch_size)
#idx, (X,_) = next(enumerate(train_loader))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

generator = GANs.Generator1(z_dim = z_dim,Data = 'CIFAR').to(device)
discriminator = GANs.Discriminator1(Data = 'CIFAR').to(device)

gen_optimizer = optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
disc_optimizer = optim.SGD(discriminator.parameters(), lr=0.0005)

EMD_log = []
D_reals = []
D_fakes = []


#fixed_noise1 = torch.randn(1, z_dim, 1, 1, device=device)
#fixed_noise2 = torch.randn(1, z_dim, 1, 1, device=device)
fixed_noise1 = torch.randn(1, z_dim, device=device).detach()
#fixed_noise2 = torch.randn(1, z_dim, device=device).detach()

count = 0

for epoch in range(150):
    for batch_idx, (real_data,label) in enumerate(train_loader):
        real_data, batch_size = sample_class(real_data,label,9)
        disc_optimizer.zero_grad()
        real_data = real_data.to(device)
        fake_data = generator.forward(GANs.sample_z(batch_size ,z_dim).to(device)).detach()   
        
#        noise_real = 0.0000001*torch.randn_like(real_data)          #noise
#        real_data += noise_real
        if batch_idx % detector_cold_time == 0:
            prediction_real = discriminator(real_data)
            loss_true_label = prediction_real.mean(0).view(1)  #1=real, 0 =fake
            loss_true_label.backward(-1*one)
            D_real = loss_true_label.item()
        
            prediction_fake = discriminator(fake_data)  
            loss_false_label = prediction_fake.mean(0).view(1)
            loss_false_label.backward(one)
            D_fake = loss_false_label.item()
            
            disc_optimizer.step()   
        
            EMD = loss_true_label - loss_false_label
            if EMD.item()>=0:
                EMD_log.append(EMD.item())
            D_reals.append(D_real)
            D_fakes.append(D_fake)
                       
        if batch_idx % 6 == 0: 
                print("current EMD*******************")
                print('epoch:'+str(epoch), 'batch:' +str(batch_idx))
                print("EMD:     % .6f" %(EMD.item()))    
                print("D_real:  % .6f " %(D_real))   
                print("D_fake: % .6f " %(D_fake))   
        #if batch_idx > 2: break
        if batch_idx % generator_cold_time == 0:        
            gen_optimizer.zero_grad() 
            gen_fake_data = generator.forward(GANs.sample_z(batch_size ,z_dim).to(device))
            gen_prediction_fake = discriminator(gen_fake_data) 
            loss_FP = gen_prediction_fake.mean(0).view(1)
            loss_FP.backward(-1*one)
            gen_optimizer.step()
            
        torch.cuda.empty_cache()
        
        if batch_idx % 100 == 0: 
            test_images1 = generator.forward(fixed_noise1.to(device))
            plt.pause(0.001) 
            plt.imshow(np.moveaxis(test_images1.cpu().detach().numpy(), 1, 3).reshape(32,32,3))
            test_images2 = generator.forward(torch.rand(1, z_dim, device=device).detach())
            plt.pause(0.001) 
            plt.imshow(np.moveaxis(test_images2.cpu().detach().numpy(), 1, 3).reshape(32,32,3))
            if  batch_idx % 100 == 0:
                save_image(test_images1.cpu(),
                         r'C:\Users\funrr\Desktop\DL3\results\cif all\a' + str(count)+ '.png')    
                count +=1   
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            ax1.plot(EMD_log, label='EMD')
            ax1.legend(prop={'size': 9})

            title = "EMD curve"
            ax1.set_title(title)
            ax1.set_xlabel("*#batchs")
            ax1.set_ylabel("EMD")
            plt.pause(0.001)
            fig1 = plt.figure()
    
            
            ax1 = fig1.add_subplot(111)
            ax1.plot(D_reals, label='D_real')
            ax1.legend(prop={'size': 9})
   
            title = "D_real curve"
            ax1.set_title(title)
            ax1.set_xlabel("*#batchs")
            ax1.set_ylabel("D_real")
            plt.pause(0.001)
           
            fig1 = plt.figure()
 
            
            ax1 = fig1.add_subplot(111)
            ax1.plot(D_fakes, label='D_fakes')
            ax1.legend(prop={'size': 9})

            title = "D_fakes curve"
            ax1.set_title(title)
            ax1.set_xlabel("*#batchs")
            ax1.set_ylabel("D_fakes")
            plt.pause(0.001)
            
            fig1
            
#----------------------savesave save save save save save save save ----------------            
for i in range(50):
    test_images2 = generator.forward(torch.rand(1, z_dim, device=device).detach())
    save_image(test_images2.cpu(),
                 r'C:\Users\funrr\Desktop\DL3\results\cif all\ship\a' + str(i)+ '.png')
    
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            ax1.plot(EMD_log, label='EMD')
            ax1.legend(prop={'size': 9})

            title = "EMD curve"
            ax1.set_title(title)
            ax1.set_xlabel("*#batchs")
            ax1.set_ylabel("JSD")
            plt.pause(0.001)
            fig1.savefig(r'C:\Users\funrr\Desktop\DL3\results\cif all'+ '/temp1.jpg',dpi = 600)  
            fig1 = plt.figure()
    
            
            ax1 = fig1.add_subplot(111)
            ax1.plot(D_reals, label='D_real')
            ax1.legend(prop={'size': 9})
   
            title = "D_real curve"W
            ax1.set_title(title)
            ax1.set_xlabel("*#batchs")
            ax1.set_ylabel("D_real")
            plt.pause(0.001)
            fig1.savefig(r'C:\Users\funrr\Desktop\DL3\results\cif all'+ '/temp2.jpg',dpi = 600)   
            fig1 = plt.figure()
 
            
            ax1 = fig1.add_subplot(111)
            ax1.plot(D_fakes, label='D_fakes')
            ax1.legend(prop={'size': 9})

            title = "D_fakes curve"
            ax1.set_title(title)
            ax1.set_xlabel("*#batchs")
            ax1.set_ylabel("D_fakes")
            plt.pause(0.001)
            fig1.savefig(r'C:\Users\funrr\Desktop\DL3\results\cif all'+ '/temp3.jpg',dpi = 600)     
            fig1