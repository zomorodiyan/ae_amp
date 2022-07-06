__author__ = 'SherlockLiao'
import numpy as np

import torch
import torch.optim
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torch.utils.tensorboard import SummaryWriter
import os
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
import matplotlib.pyplot as plt

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)

    x = x.view(x.size(0), 1, 28, 28)
    return x


num_epochs = 1
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

dataset = MNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
model, optimizer = amp.initialize(model, optimizer, opt_level="O2",loss_scale=1)

loss_values = []


writer = SummaryWriter('/runs/cae')
step=0
for epoch in range(num_epochs):
    for i,data in enumerate(dataloader):
        img, _ = data
        img = Variable(img).cuda()
        # ---- forward -------------------------------------------------------
        output = model(img)
        loss = criterion(output, img)
        print('loss:',loss.item())
        # ---- backward ------------------------------------------------------
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        # ---- log -----------------------------------------------------------
        writer.add_scalar('training_loss',loss.item(),global_step=step)
        writer.add_scalar('en_conv0.bias_0',model.encoder[0].bias[0],global_step=step)
        writer.add_scalar('de_conv0.bias_0',model.decoder[0].bias[0],global_step=step)
        writer.add_histogram('en_conv0.weight',model.encoder[0].weight,global_step=step)
        writer.add_histogram('en_conv1.weight',model.encoder[3].weight,global_step=step)
        writer.add_histogram('de_conv0.weight',model.decoder[0].weight,global_step=step)
        writer.add_histogram('de_conv1.weight',model.decoder[2].weight,global_step=step)
        writer.add_histogram('de_conv2.weight',model.decoder[4].weight,global_step=step)
        step+=1
writer.close()

#torch.save(model.state_dict(), './conv_autoencoder.pth')
