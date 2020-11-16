import numpy as np
import pandas as pd

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
epochs = 10
learning_rate = 1e-4
bottleneck_size = 2
encoder_output_size = 512
decoder_output_size = 512

torch.set_default_tensor_type('torch.DoubleTensor')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """
    Define an encoder network architecture
    to project original data to a lower dimension
    """
    def __init__(self, input_dimensions):
        super(Encoder, self).__init__() 
        self.input_dim = input_dimensions
        self.input_layer = nn.Linear(in_features=self.input_dim, out_features=encoder_output_size).to(DEVICE)
        self.relu = nn.ReLU(True)
        self.encoder_h1 = nn.Linear(encoder_output_size, 256).to(DEVICE)
        self.encoder_h2 = nn.Linear(256, 128).to(DEVICE)
        self.bottleneck_layer = nn.Linear(128, 64).to(DEVICE)

    def forward(self, x):
        x = x.to(DEVICE)
        input_o = self.input_layer(x)
        input_relu = self.relu(input_o)
        h1_o = self.encoder_h1(input_relu)
        h1_relu = self.relu(h1_o)
        h2_o = self.encoder_h2(h1_relu)
        h2_relu = self.relu(h2_o)
        encoded_o = self.bottleneck_layer(h2_relu)
        return encoded_o


class Decoder(nn.Module):
    """
    Define a decoder network architecture 
    to reconstruct original data from its lower dimensional representation
    """
    def __init__(self, input_dimensions):
        super(Decoder, self).__init__()

        self.input_dim = input_dimensions
        self.bottleneck_layer = nn.Linear(in_features=64, out_features=128)
        self.relu = nn.ReLU(True)
        self.decoder_h1 = nn.Linear(128, 256)
        self.decoder_h2 = nn.Linear(256, 512)
        self.decoded_layer = nn.Linear(512, self.input_dim)

    def forward(self, x):
        x = x.to(DEVICE)
        decoder_b = self.bottleneck_layer(x)
        b_relu = self.relu(decoder_b)
        h1_o = self.decoder_h1(b_relu)
        h1_relu = self.relu(h1_o)
        h2_o = self.decoder_h2(h1_relu)
        h2_relu = self.relu(h2_o)
        decoded_o = self.decoded_layer(h2_relu)
        return decoded_o


class SCAutoEncoder(nn.Module):
    """
    Merge encoder and decoder networks
    """
    def __init__(self, **kwargs):
        super(SCAutoEncoder, self).__init__()
        self.input_dim = kwargs["input_dim"]
        self.encoder = Encoder(self.input_dim)
        self.decoder = Decoder(self.input_dim)

    def forward(self, features):
        features = features.to(DEVICE)
        encoded_features = self.encoder(features)
        reconstructed_features = self.decoder(encoded_features)
        return reconstructed_features

    def train_model(self, input_data, test_data, sc_tr_data, sc_te_data):
        dataloader = DataLoader(input_data, batch_size=batch_size, shuffle=True)
        model = SCAutoEncoder(input_dim=self.input_dim).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # mean-squared error loss
        criterion = nn.MSELoss()

        print("Start training...")
        for epoch in range(epochs):
            loss = 0
            for batch_features in dataloader:
                optimizer.zero_grad()
                batch_features = batch_features.to(DEVICE)
                outputs = model(batch_features).to(DEVICE)
                outputs = outputs.to(DEVICE)
                train_loss = criterion(outputs, batch_features).to(DEVICE)
                train_loss.backward()
                optimizer.step()
                loss += train_loss.item()
            loss = loss / len(dataloader)
            print("epoch : {}/{}, loss = {:.4f}".format(epoch + 1, epochs, loss))
        # load test data
        test_loader = DataLoader(test_data, batch_size=test_data.shape[0], shuffle=True)        
        sc_pp = post_processing.SCPostProcessing(sc_te_data, output_file="data/output.csv", sc_test_file="data/sc_test_file.h5ad")
        for te_d in test_loader:
            p_data = self.encoder.forward(torch.tensor(np.array(te_d)).to(DEVICE))
            sc_pp.save_results(p_data.detach().tolist())
