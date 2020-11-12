import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision


seed = 42
torch.manual_seed(seed)
#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.deterministic = True

batch_size = 32
epochs = 10
learning_rate = 1e-4
bottleneck_size = 2
encoder_output_size = 32
decoder_output_size = 32

torch.set_default_tensor_type('torch.DoubleTensor')


class Encoder(nn.Module):
    """
    Define an encoder network architecture 
    to project original data to a lower dimension
    """
    def __init__(self, input_dimensions):
        super(Encoder, self).__init__()
        
        self.input_dim = input_dimensions
        self.input_layer = nn.Linear(in_features=self.input_dim, out_features=encoder_output_size)
        self.relu = nn.ReLU(True)
        self.encoder_h1 = nn.Linear(encoder_output_size, 16)
        self.encoder_h2 = nn.Linear(16, 8)
        self.bottleneck_layer = nn.Linear(8, 2)
        
    def forward(self, x):
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
        self.bottleneck_layer = nn.Linear(in_features=2, out_features=8)
        self.relu = nn.ReLU(True)
        self.decoder_h1 = nn.Linear(8, 16)
        self.decoder_h2 = nn.Linear(16, 32)
        self.decoded_layer = nn.Linear(32, self.input_dim)

    def forward(self, x):
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
        encoded_features = self.encoder(features)
        reconstructed_features = self.decoder(encoded_features)
        return reconstructed_features

    def train_model(self, input_data, test_data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = DataLoader(input_data, batch_size=batch_size, shuffle=True)

        model = SCAutoEncoder(input_dim=self.input_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # mean-squared error loss
        criterion = nn.MSELoss()

        print("Start training...")
        for epoch in range(epochs):
            loss = 0
            for batch_features in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_features)
                train_loss = criterion(outputs, batch_features)
                train_loss.backward()
                optimizer.step()
                loss += train_loss.item()
            loss = loss / len(dataloader)
            print("epoch : {}/{}, loss = {:.4f}".format(epoch + 1, epochs, loss))
        # load test data
        test_loader = DataLoader(test_data, batch_size=test_data.shape[0], shuffle=True)
        for te_d in test_loader:
            p_data = self.encoder.forward(torch.tensor(np.array(te_d)))
            print(p_data)
            self.save_results(p_data)
            
    def save_results(self, pred_results, output_file="data/output.csv"):
        res_np = pred_results.detach().numpy()
        dataframe = pd.DataFrame(res_np)
        dataframe.to_csv(output_file, sep="\t", header=False, index=False, index_label=False)
        #np.savetxt(output_file, pred_results.detach().numpy(), delimiter='')
