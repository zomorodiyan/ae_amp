import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm as tqdm # progress-bar for loops

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
#from torchvision.datasets import MNIST
from torch.utils.tensorboard import SummaryWriter
from torch.cuda import profiler
from torch.utils.data import Dataset


from torchsummary import summary

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("install apex from https://www.github.com/nvidia/apex")


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epochs = 100
batch_size = 128
learning_rate = 1e-5

'''
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
dataset = MNIST('./data', transform=img_transform, download=True)
'''


#%%
f = h5py.File('sst_weekly.mat','r') # can be downloaded from https://drive.google.com/drive/folders/1pVW4epkeHkT2WHZB7Dym5IURcfOP4cXu?usp=sharing
lat = np.array(f['lat'])
lon = np.array(f['lon'])
sst = np.array(f['sst'])

sst2 = np.zeros((len(sst[:,0]),len(lat[0,:]),len(lon[0,:])))
for i in tqdm(range(len(sst[:,0]))):
    sst2[i,:,:] = np.flipud((sst[i,:].reshape(len(lat[0,:]),len(lon[0,:]),order='F')))


#sst_no_nan = np.nan_to_num(sst)
sst = sst.T
num_samples = sst.shape[1]

for i in range(num_samples):
    nan_array = np.isnan(sst[:,i])
    not_nan_array = ~ nan_array
    array2 = sst[:,i][not_nan_array]
    if i == 0:
        num_points = array2.shape[0]
        sst_masked = np.zeros((array2.shape[0],num_samples))
    sst_masked[:,i] = array2

#%%
train_end = 1500
u = sst_masked[:,:train_end]
utest  = sst_masked[:,train_end:]

u_small = np.zeros((int(u.shape[0]/10),u.shape[1]))
for i in range(int(u.shape[0]/10)):
  for j in range(u.shape[1]):
    u_small[i,j] = u[10*i,j]



class sstDataset(Dataset):
  def __init__(self):
    # data loading
    self.x = u_small/18
    self.y = u_small/18
    self.n_samples = u_small.shape[1]

  def __getitem__(self, index):
    return self.x[:,index], self.y[:,index]

  def __len__(self):
    return self.n_samples

dataset = sstDataset()
xx,yy = dataset[0]
print('xxx,yyy')
print(xx[:3],yy[:3])

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4421, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 100),
            nn.ReLU(True),
            nn.Linear(100, 40),
            nn.ReLU(True),
            nn.Linear(40, 30))
        self.decoder = nn.Sequential(
            nn.Linear(30, 40),
            nn.ReLU(True),
            nn.Linear(40, 100),
            nn.ReLU(True),
            nn.Linear(100, 1000),
            nn.ReLU(True), nn.Linear(1000, 4421))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder().cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
model, optimizer = amp.initialize(model, optimizer,
                                  opt_level="O1",loss_scale=0.01)

loss_values = []

=======
# ~~~~ tensorboard ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Creates a file writer for the log directory.
logdir = "logs/cae/O0"
writer = SummaryWriter(logdir)

# Sets up a timestamped logs.
global_step=0
for epoch in range(num_epochs):
    for i,data in enumerate(dataloader):
        img, _ = data
        img = Variable(img).cuda()

        # ~~~~ forward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        output = model(img)
        loss = criterion(output, img)
        print('loss:',loss.item())

        # ~~~~ backward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        # ~~~~ tensorboard ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        '''
        def log_scale(a):
          #return np.log10(np.abs(a.cpu().detach().numpy()))
          return np.log10(np.abs(a.cpu().numpy()))

        convolution_layers = {
          "En0":model.encoder[0], "En1":model.encoder[3],
          "De0":model.decoder[0], "De1":model.decoder[2], "De2":model.decoder[4],}

        for layer_name in convolution_layers:
          #weight = convolution_layers[layer_name].weight
          weightGrad = convolution_layers[layer_name].weight.grad
          #bias = convolution_layers[layer_name].bias
          #biasGrad = convolution_layers[layer_name].bias.grad

          writer.add_histogram(layer_name+"_WGrad",weightGrad,global_step=global_step)
          writer.add_histogram(layer_name+"_WGrad_tensorflow_bins",weightGrad,global_step=global_step,bins='fd')
          writer.add_histogram(layer_name+"_WGrad_Log",log_scale(weightGrad),global_step=global_step)
        '''

        writer.add_scalar('training_loss',loss.item(),global_step=global_step)
        global_step+=1
writer.close()


'''
fig,axs = plt.subplots(1,1, figsize=(10,8))
current_cmap = plt.cm.get_cmap('RdYlBu')
current_cmap.set_bad(color='black',alpha=0.8)
cs = axs.imshow(sst2[0,:,:],cmap='RdYlBu')
# axs.grid()
fig.colorbar(cs, ax=axs, orientation='vertical',shrink=0.4)
fig.tight_layout()
plt.savefig('img.png')
'''
