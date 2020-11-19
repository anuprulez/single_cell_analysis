import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

import post_processing


seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

batch_size = 32
num_epochs = 1
learning_rate = 0.0002
z_dim = 128
beta1 = 0.5
ngpu = 0

torch.set_default_tensor_type('torch.DoubleTensor')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Discriminator(nn.Module):
    """
    Define an encoder network architecture
    to project original data to a lower dimension
    """
    def __init__(self, ngpu, input_dimensions):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
             nn.Linear(in_features=input_dimensions, out_features=1024),
             nn.ReLU(True),
             nn.Linear(in_features=1024, out_features=512),
             nn.ReLU(True),
             nn.Linear(in_features=512, out_features=256),
             nn.ReLU(True),
             nn.Linear(in_features=256, out_features=z_dim),
             nn.ReLU(True),
             nn.Linear(in_features=z_dim, out_features=1),
             nn.Sigmoid()
        )

    def forward(self, input):
        return self.discriminator(input)        
        

class Generator(nn.Module):
    """
    Define an encoder network architecture
    to project original data to a lower dimension
    """
    def __init__(self, ngpu, input_dimensions):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=256),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(True),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(True),
            nn.Linear(in_features=1024, out_features=input_dimensions),
            nn.Tanh()
        )

    def forward(self, input):
        return self.generator(input)  
        
    
class GAN(object):
    """
    
    """
    
    def __init__(self, input_dim):
        super(GAN, self).__init__()
        self.input_dim = input_dim

    def trainGAN(self, input_data, test_data, sc_train_data, sc_test_data):
        dataloader = DataLoader(input_data, batch_size=batch_size, shuffle=True)

        # Create the generator
        netG = Generator(ngpu, self.input_dim).to(DEVICE)
        # Print the model
        print(netG)

        # Create the Discriminator
        netD = Discriminator(ngpu, self.input_dim).to(DEVICE)
         # Print the model
        print(netD)

        # Initialize BCELoss function
        criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(batch_size, z_dim, 1, 1, device=DEVICE)

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

        # Training Loop

        # Lists to keep track of progress
        G_losses = []
        D_losses = []

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                netD.zero_grad()
                # Format batch
                real_cpu = data.to(DEVICE)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.double, device=DEVICE)
                # Forward pass real batch through D
                output = netD(real_cpu).view(-1)
                
                errD_real = criterion(output, label).to(DEVICE)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                
                # Generate batch of latent vectors
                noise = torch.randn(batch_size, z_dim, device=DEVICE)

                # Generate fake image batch with G
                fake = netG(noise)
                
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()
              
                # Output training stats
                if i % 2 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            '''if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1'''
        plot_losses(G_losses, D_losses)
            
    def plot_losses(self, G_losses, D_losses):
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses,label="G")
        plt.plot(D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()    
