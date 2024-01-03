#NHGD
import torch
import os
import torch
from torch import norm
from torch import autograd
from torch import nn
import scipy
from scipy.sparse.linalg import cg
from scipy.sparse.linalg.interface import LinearOperator as LO
import numpy as np
import gc
import time
from decimal import *
import matplotlib.pyplot as plt

# Hyperparameters of the optimizer
lr = 1e-5
lam = 1e-2
# Batch size during training
batchsize = 64
# The gradient penalty hyperparameter
gp_lam = 1
dirname = output_dir = 'output2'

def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.detach()
            p.grad.zero_()




import torch
import torch.autograd as autograd
import numpy as np
class NHGD:

  def __init__(self, min_params, max_params, lr, lam):
    self.min_params = list(min_params)
    self.max_params = list(max_params)
    self.lr = lr
    self.lam = lam#torch.tensor(lam, device='cuda')
    self.state = {}
    self.state['lam'] = self.lam
    self.state['gradx_norm'] = []
    self.state['grady_norm'] = []
    self.moment = 0
    
    
    self.state['H_xx_grad_x_norm'] = []
    self.state['H_yx_grad_y_norm'] = []
    self.state['H_yy_grad_y_norm'] = []
    self.state['H_xy_grad_x_norm'] = []
    self.previous = 0
  
  def step(self, disc_loss, gen_loss):

        
    max_params = list(self.max_params)
    min_params = list(self.min_params)
    params = min_params + max_params
    
    grad_min = autograd.grad(disc_loss, self.min_params, create_graph=True, retain_graph=True) # Min params
    grad_max = autograd.grad(gen_loss, self.max_params, create_graph=True, retain_graph=True) # Max params

    grad_min_vec = torch.cat([g.contiguous().view(-1) for g in grad_min]) # M x 1
    grad_max_vec = torch.cat([g.contiguous().view(-1) for g in grad_max]) # N x 1
    
    
    
    
    grads_vec = torch.cat((grad_min_vec.detach(), grad_max_vec.detach()),0)
    grads_vec_d = grads_vec.detach()
    
    self.moment = 0.99 * self.moment + (1-0.99) * grads_vec_d * grads_vec_d
    grads_vec_d = grads_vec_d/ (torch.sqrt(self.moment) + 1e-7)
    grads_vec_d_flatten = grads_vec_d.reshape(-1,1)
#
    
    del_op =  grads_vec_d.reshape(-1,1)
    self.lr = self.lr
    
    gH = np.sqrt(self.lr) * del_op.reshape(-1,1)
    
    sqrt_lam = float(np.sqrt(self.lam))
    v = gH / sqrt_lam
    u = gH

    IT = v.T @ v


    sol = 1/sqrt_lam * (grads_vec_d_flatten  - (v @ (v.T@grads_vec_d_flatten)/(1+IT)))

    term = -1  * (grads_vec_d.reshape(-1,1) -  sol.reshape(-1,1))
    index=0
    for p in self.min_params:
        #print(p.numel(), index)
        p.data.add_(self.lr * term[index:index+p.numel()].reshape(p.shape))
        index += p.numel()

    for p in self.max_params:
        #print(p.numel(), index)
        p.data.add_(self.lr * term[index:index+p.numel()].reshape(p.shape))
        index += p.numel()
    self.previous = del_op
    del del_op
    del grad_min
    del grad_max
import numpy as np
def plotter(data, title, filename):
    plt.clf()
    plt.plot(data)
    plt.title(title)
    plt.xlabel("batches")
    #plt.xticks(np.arange(len(data)))
    plt.savefig(filename)
    
    
    
    
from torch import Tensor
from torch.autograd import Variable
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device='cuda')
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates).reshape(-1,1)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0).to('cuda'), requires_grad=False)
    # Get gradient w.r.t. interpolates
    #print(d_interpolates.shape, interpolates.shape, fake.shape)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


#from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
#from IPython.display import HTML
import torchvision


import csv
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models.inception import inception_v3
import torchvision.utils as vutils


n_features = 80
latent_space = 80
num_eig = 2
reg = 1

maxiter = 5

z_dim = 100
device = 'cuda'
noise_shape=(64, 100, 1, 1)

manualSeed = 999

random.seed(manualSeed)
torch.manual_seed(manualSeed)


dataroot = "../"




dataset = CIFAR10(root='./data', train=True,
                          transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                        ]),
                          download=True)
nc=3


# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize,
                                        shuffle=True, drop_last=True)



# Decide which device we want to run on
device = "cuda"


DIM = 64
class GoodDiscriminator(torch.nn.Module):
    def __init__(self, channels=3):
        super(GoodDiscriminator, self).__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True))
            # outptut of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0),
            # Output 1
            # cause changed loss to BCEWithLogitsLoss from BCELoss
            # nn.Sigmoid()
            )

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384 features
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)
    
class GoodGenerator(torch.nn.Module):
    def __init__(self, channels=3):
        # import ipdb; ipdb.set_trace()
        super(GoodGenerator, self).__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            # Z latent vector 100
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),

            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=channels, kernel_size=4, stride=2, padding=1))
            # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)
    
netD = GoodDiscriminator().to('cuda')
netG = GoodGenerator().to('cuda')


def weights_init_d(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.005)

def weights_init_g(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.005)
      

netD.apply(weights_init_d)
netG.apply(weights_init_g)

def generate_data():
        global batchsize, z_dim, device, netG
        z = torch.randn((batchsize, z_dim, 1, 1), device=device)
        data = netG(z)
        return data

def weights_init_g(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.005)
      


        
# Initialize BCELoss function
criterion = nn.BCEWithLogitsLoss()
fixed_noise = torch.randn(noise_shape, device=device)

real_label = 1.
fake_label = 0.


from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def plotter(data, title, xlabel,ylabel, filename):
    figure(num=None, figsize=(10,10),dpi=80, facecolor='w', edgecolor='k')
    epoch = len(data)
    y = np.linspace(1, epoch, epoch)
    plt.title(title)

    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)


    plt.xscale('log')
    plt.plot(data, color='k', )#markevery=1)
    plt.savefig(filename+'_log'+'.png')
    plt.xscale('linear')
    plt.plot(data, color='k')
    plt.savefig(filename+'_linear'+'.png')
    plt.clf()
    plt.close()

# Training Loop

device = 'cuda'
optimizer = NHGD(max_params = netG.parameters(), min_params=netD.parameters(), lr=lr, lam=lam,)


import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy


def gradient_penalty_calculator(real_x, fake_x):
        global gp_lam, netD, batch_size
        
        alpha = torch.randn((batchsize, 1, 1, 1), device=device)
        #print(real_x.shape, fake_x.shape, alpha.shape)
        alpha = alpha.expand_as(real_x)
        interploted = alpha * real_x.data + (1.0 - alpha) * fake_x.data
        interploted.requires_grad = True
        interploted_d = netD(interploted)
        gradients = torch.autograd.grad(outputs=interploted_d, inputs=interploted,
                                        grad_outputs=torch.ones(interploted_d.size(),
                                                                device=device),
                                        create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(batchsize, -1)
        
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return gp_lam * ((gradients_norm - 1.0) ** 2).mean()

def get_inception_score(batch_num, splits_num=10):
        global batchsize, generate_data
        net = inception_v3(pretrained=True, transform_input=False).eval().to(device)
        resize_module = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True).to(
            device)
        preds = np.zeros((batchsize * batch_num, 1000))
        for e in range(batch_num):
            imgs = resize_module(generate_data())
            pred = F.softmax(net(imgs), dim=1).data.cpu().numpy()
            preds[e * batchsize: e * batchsize + batchsize] = pred
        split_score = []
        chunk_size = preds.shape[0] // splits_num
        for k in range(splits_num):
            pred_chunk = preds[k * chunk_size: k * chunk_size + chunk_size, :]
            kl_score = pred_chunk * (
                        np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
            kl_score = np.mean(np.sum(kl_score, 1))
            split_score.append(np.exp(kl_score))
        return np.mean(split_score), np.std(split_score)




import os



def write_rows(gen_loss = None, disc_loss = None):

  if(gen_loss == None and disc_loss == None):
    global writer
    global csv_file
    csv_file = open(output_dir+'/plots/' + 'loss_data.csv', 'w')
    writer = csv.DictWriter(csv_file, fieldnames = ['gen_loss', 'disc_loss'])
    writer.writeheader()
    csv_file.flush()

  elif((gen_loss == None and disc_loss != None) or (gen_loss != None and disc_loss == None)):
    raise "Either generator's or discriminator's loss is NULL"
  else:
    dict_data = {'gen_loss' : str(gen_loss),
                      'disc_loss' : str(disc_loss) }
    writer.writerow(dict_data)
    csv_file.flush()

write_rows()

epoch_num = 100000

d_iter = 1

def transform(x):
    x = transforms.ToTensor()(x)
    return (x - 0.5) / 0.5


def detransform(x):
    return (x + 1.0) / 2.0

import time
start = time.time()
timer = time.time()
device='cuda'
count = 0
show_iter = len(dataloader)//batchsize
startPoint = 0


import csv

writer_file = open(output_dir + "inception.csv", "w")
writer = csv.writer(writer_file)
writer.writerow(['inception_scores_mean', 'inception_scores_std', 'time_per_update', 'total_time'])
import time
total_time_tic = time.time()

if(os.path.isfile(output_dir + '/netG')):
    netG.load_state_dict(torch.load(output_dir + '/netG'))
    netD.load_state_dict(torch.load(output_dir + '/netD'))



best_inception = 0
for e in range(epoch_num):
    update_time_sum = 0
    for real_x in dataloader:
        real_x = real_x[0].to(device)
        fake_x = generate_data()
        d_real = netD(real_x)
        d_fake = netD(fake_x)
        d_loss = d_fake.mean() - d_real.mean()
        
        gradient_penalty = gradient_penalty_calculator(real_x=real_x, fake_x=fake_x)
        d_loss = d_loss + gradient_penalty
        g_loss = - d_fake.mean()
        update_time_tic = time.time()
        optimizer.step(d_loss, g_loss)
        update_time_toc = time.time()
        update_time_sum = update_time_sum + update_time_toc - update_time_tic
        timer = time.time()
            
        #if count % show_iter == 0:
        timer = time.time() - timer
        
        img = netG(fixed_noise).detach()
        img = detransform(img)
        
        path = 'figs/%s/' % dirname
        if not os.path.exists(path):
            os.makedirs(path)
        
        timer = time.time()
        count += 1
    if not os.path.exists(path):
            os.makedirs(path)
            
            timer = time.time()
    vutils.save_image(img, path + 'bniter_%d.png' % (e))
            
#if count % show_iter == 0:
    with torch.no_grad():

        
        inception_score = get_inception_score(batch_num=500)
        inception_score_mean, inception_scores_std = inception_score
        if(inception_score_mean > best_inception):
            best_inception = inception_score_mean
            torch.save(netG.state_dict(),  output_dir + 'generator.pkl')
            torch.save(netD.state_dict(), output_dir + 'discriminator.pkl')
            vutils.save_image(img, path +  output_dir+'best_inception.png' )
        np.set_printoptions(precision=4)
        
        print('inception score mean: {}, std: {}'.format(inception_score[0], inception_score[1]))
        total_time_toc = time.time()
        writer.writerow((inception_score_mean, inception_scores_std, update_time_sum/batchsize, total_time_toc - total_time_tic))
        writer_file.flush()

            
            
            
            
