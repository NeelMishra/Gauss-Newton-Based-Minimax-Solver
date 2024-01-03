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
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from IPython.display import HTML
import torchvision
import csv
lr=0.02
lam= 1e-1
gp_lam = 1e-1
# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
#print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)



dataroot = "../"

#Output directory

output_dir = 'output1'

#Maximum number of CG iters

maxiter = 4000

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 17

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 28

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 28

# Size of feature maps in discriminator
ndf = 28

# Number of training epochs
num_epochs = 10000


# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1



# We can use an image folder dataset the way we have it setup.
# Create the dataset


dataset = torchvision.datasets.MNIST(root=dataroot, transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]), download= True)

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers, drop_last=True)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
#real_batch = next(iter(dataloader))
#plt.figure(figsize=(8,8))
#plt.axis("off")
#plt.title("Training Images")
#plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



# Generator Code

class Generator(nn.Module):
    def __init__(self, nc, nz, ngf):
      super(Generator, self).__init__()
      self.network = nn.Sequential(
          nn.ConvTranspose2d(nz, ngf*4, 4, 1, 0, bias=False),
          nn.BatchNorm2d(ngf*4),
          nn.ReLU(True),
  
          nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, bias=False),
          nn.BatchNorm2d(ngf*2),
          nn.ReLU(True),
  
          nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
          nn.BatchNorm2d(ngf),
          nn.ReLU(True),
  
          nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
          nn.Tanh()
      )


    def forward(self, input):
      output = self.network(input)
      return output



netG = Generator(nc,nz,ngf).to(device)

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
#netG.apply(weights_init)

# Print the model
#print(netG)



class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
                
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
    def forward(self, input):
        output = self.network(input)
        return output.view(-1, 1).squeeze(1)


# Create the Discriminator
netD = Discriminator(nc,ndf).to(device)




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
    
    
    
    
    grads_vec = torch.cat((grad_min_vec, grad_max_vec),0)
    grads_vec_d = grads_vec.clone().detach()
    
    
    del_op =  grads_vec_d.reshape(-1,1)
    gH = np.sqrt(self.lr) * del_op.reshape(-1,1)
    
    sqrt_lam = float(np.sqrt(self.lam))
    v = gH / sqrt_lam
    u = gH

    IT = v.T @ v
    
    xx_reg = grad_max_vec.detach().T @ grad_max_vec.detach()
    yy_reg = grad_min_vec.detach().T @ grad_min_vec.detach()
    

    sol = 1/sqrt_lam * (del_op  - (v @ IT/(1+IT)))


  
    term = -1 * self.lr * (grads_vec.reshape(-1,1) -  sol.reshape(-1,1))
    index=0
    for p in self.min_params:
        #print(p.numel(), index)
        p.data.add_(term[index:index+p.numel()].reshape(p.shape))
        index += p.numel()

    for p in self.max_params:
        #print(p.numel(), index)
        p.data.add_(term[index:index+p.numel()].reshape(p.shape))
        index += p.numel()
    self.previous = del_op
device = 'cuda'
optimizer = NHGD(max_params = netG.parameters(), min_params=netD.parameters(), lr=lr, lam=lam,)



# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.


# Print the model
#print(netD)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
#optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
#optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

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

# Lists to keep track of progress
iter_list = []
time_list = []
res_list = []
disc_loss = []
gen_loss = []
iters = 0
device = 'cuda'


#print("Starting Training Loop...")
# For each epoch
iter_list = []
time_list = []
res_list = []
disc_loss = []
gen_loss = []
#maxiter=5000
#output_dir = 'output1'
#iters = 0
#num_epochs = 10000
device = 'cuda'


import os


writer_file = open(output_dir + "loss.csv", "w")
writer = csv.writer(writer_file)
writer.writerow(['gen_loss', 'disc_loss'])

#print("Starting Training Loop...")
# For each epoch


def gradient_penalty_calculator(real_x, fake_x):
        global gp_lam, netD, batch_size
        
        alpha = torch.randn((batch_size, 1, 28, 28), device=device)
        #print(real_x.shape, fake_x.shape, alpha.shape)
        alpha = alpha.expand_as(real_x)
        interploted = alpha * real_x.data + (1.0 - alpha) * fake_x.data
        interploted.requires_grad = True
        interploted_d = netD(interploted)
        gradients = torch.autograd.grad(outputs=interploted_d, inputs=interploted,
                                        grad_outputs=torch.ones(interploted_d.size(),
                                                                device=device),
                                        create_graph=True, retain_graph=True)[0]
        gradients = gradients.view(batch_size, -1)
        
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return gp_lam * ((gradients_norm - 1.0) ** 2).mean()

for epoch in range(num_epochs):
    # For each batch in the dataloader
    avg_gen_loss_sum_per_epoch = 0
    avg_disc_loss_sum_per_epoch = 0
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        #netD.zero_grad()
        # Format batch
        real_x = data[0].to(device)
        b_size = real_x.size(0)
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        
        #label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        d_real = output = netD(real_x)
        d_fake = netD(fake)
        d_loss = d_fake.mean() - d_real.mean()
        gradient_penalty = gradient_penalty_calculator(real_x=real_x, fake_x=fake)
        d_loss = d_loss + gradient_penalty
        g_loss = - d_fake.mean()
        
        optimizer.step(d_loss, g_loss)
        avg_gen_loss_sum_per_epoch += g_loss.detach().item()
        avg_disc_loss_sum_per_epoch += d_loss.detach().item()
        # Output training starts
        
            
            

    writer.writerow((avg_gen_loss_sum_per_epoch/len(dataloader), avg_disc_loss_sum_per_epoch/len(dataloader)))
    writer_file.flush()
    with torch.no_grad():
        #old = optimizer.old.clone()
        #del optimizer
        #optimizer = EQ5(max_params = netG.parameters(), min_params=netD.parameters(), lr_max=lr,
        #         lr_min=lr,device = device, alpha = 0,maxiter=maxiter, tol=1e-12, old = old )

        
        fake = netG(fixed_noise).detach().cpu()
        plt.imsave(output_dir + '/plot_'+str(epoch) +'.png',np.transpose(vutils.make_grid(fake, padding=2, normalize=True).numpy(),(1,2,0)))
        #img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        torch.save(netG.state_dict(), output_dir + '/netG')
        torch.save(netD.state_dict(), output_dir + '/netD')
        iters += 1



