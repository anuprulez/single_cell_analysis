import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

batch_size = 512
epochs = 20
learning_rate = 1e-3


class SCAutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.sigmoid(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.sigmoid(activation)
        return reconstructed
        
    def setup_training(input_shape):
        #  use gpu if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # create a model from `AE` autoencoder class
        # load it to the specified device, either gpu or cpu
        model = SCAutoEncoder(input_shape=input_shape).to(device)

        # create an optimizer object
        # Adam optimizer with learning rate 1e-3
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # mean-squared error loss
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            loss = 0
            for batch_features, _ in train_loader:
                # reshape mini-batch data to [N, 784] matrix
                # load it to the active device
                batch_features = batch_features.view(-1, input_shape).to(device)
        
                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                optimizer.zero_grad()
        
                # compute reconstructions
                outputs = model(batch_features)
        
                # compute training reconstruction loss
                train_loss = criterion(outputs, batch_features)
        
                # compute accumulated gradients
                train_loss.backward()
        
                # perform parameter update based on current gradients
                optimizer.step()
        
                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()
    
            # compute the epoch training loss
            loss = loss / len(train_loader)
    
            # display the epoch training loss
            print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))
