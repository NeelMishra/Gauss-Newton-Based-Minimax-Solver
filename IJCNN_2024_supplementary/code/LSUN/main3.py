
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from torch import nn
import csv
import torchvision.utils as vutils
from torchvision.datasets import LSUN

show_iter = 1000
lsun_class = '' # Define the lsun class here, example for lsun bride it is 'bridge_train', refer to pytorch documentation of torchvision.datasets.LSUN
dirname = output_dir = 'output3'
lr = 1e-5
lam = 1e-2
batchsize = 64
gp_lam = 1
show_iter = 1000
data_url = '' # Path to the dataset
test_data_class10_dir = os.path.join(os.getcwd(),data_url)

n_features = 80
latent_space = 80
num_eig = 2
reg = 1

maxiter = 5

z_dim = 100
device = 'cuda'
noise_shape=(64, 100, 1, 1)


''' BATCHLOADER '''
def get_data_loader(data_dir= test_data_class10_dir, batch_size=batchsize, train = True):
    """
    Define the way we compose the batch dataset including the augmentation for increasing the number of data
    and return the augmented batch-dataset
    :param data_dir: root directory where the either train or test dataset is
    :param batch_size: size of the batch
    :param train: true if current phase is training, else false
    :return: augmented batch dataset
    """

    # define how we augment the data for composing the batch-dataset in train and test step
    transform = {
        'train': transforms.Compose([
            transforms.Resize([64,64]),
            transforms.RandomHorizontalFlip(), # Flip the data horizontally
            #TODO if it is needed, add the random crop
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ]),
        'test': transforms.Compose([
            transforms.Resize([64,64]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
    }

    # ImageFloder with root directory and defined transformation methods for batch as well as data augmentation
    data = torchvision.datasets.ImageFolder(root=data_dir, transform=transform['train'] if train else 'test')
    data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

    return data_loader
transform = {
        'train': transforms.Compose([
            transforms.Resize([64,64]),
            transforms.RandomHorizontalFlip(), # Flip the data horizontally
            #TODO if it is needed, add the random crop
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ]),
        'test': transforms.Compose([
            transforms.Resize([64,64]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
    }
#data_loader = get_data_loader()
data_loader = LSUN(root=data_url, classes = [lsun_class], transform=transform["train"])
from torch.utils.data import DataLoader

data_loader = DataLoader(data_loader, batch_size=64, shuffle=True, drop_last=True)
''''''
## The way to get one batch from the data_loader
#if __name__ == "__main__":
#    torch.multiprocessing.freeze_support()
#    data_loader = get_data_loader()
#
#    for i in range(len(data_loader)):
#        batch_x, batch_y = next(iter(data_loader))
#        print(np.shape(batch_x), batch_y)

z_dim = 100

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
            nn.LeakyReLU(0.2, inplace=True),
            # outptut of main module --> State (1024x4x4)
            
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2, inplace=True),)
            # outptut of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=1, kernel_size=4, stride=1, padding=0),
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

#             # State (256x16x16)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),
#             # output of main module --> Image (Cx32x32)
            
            # State (128x32x32)
            nn.ConvTranspose2d(in_channels=128, out_channels=channels, kernel_size=4, stride=2, padding=1)
            )
            # output of main module --> Image (Cx64x64)

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







# Initialize BCELoss function
criterion = nn.BCEWithLogitsLoss()
fixed_noise = torch.randn(noise_shape, device=device)
# Create batch of latent vectors that we will use to visualize
#  the progression of the generator

# Establish convention for real and fake labels during training
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


import os

if(os.path.isfile(output_dir + '/netG')):
    netG.load_state_dict(torch.load(output_dir + '/netG'))
    netD.load_state_dict(torch.load(output_dir + '/netD'))

    
    

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

startPoint = 0


import csv
epoch_num = 100000000

for e in range(epoch_num):
    print(e)
    for i,real_x in enumerate(data_loader):
    
        print("INSIDE ", i)
        real_x = real_x[0].to(device)
        fake_x = generate_data()
        d_real = netD(real_x)
        d_fake = netD(fake_x)
        d_loss = d_fake.mean() - d_real.mean()
        
        gradient_penalty = gradient_penalty_calculator(real_x=real_x, fake_x=fake_x)
        d_loss = d_loss + gradient_penalty
        g_loss = - d_fake.mean()

        optimizer.step(d_loss, g_loss)

        timer = time.time()
            
        if count % show_iter == 0:
            timer = time.time() - timer
            
            img = netG(fixed_noise).detach()
            img = detransform(img)
            
            path = 'figs/%s/' % dirname
            if not os.path.exists(path):
                os.makedirs(path)
            vutils.save_image(img, path +"EPOCH_" + str(e) + '_bniter_%d.png' % (count + startPoint))
            timer = time.time()
            
            if not os.path.exists(path):
                    os.makedirs(path)
                    vutils.save_image(img, path + 'bniter_%d.png' % (count + startPoint))
                    timer = time.time()
        
        count += 1
    torch.save(netG.state_dict(),  output_dir + '_generator.pkl')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
     
