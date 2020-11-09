import numpy as np

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

class SCAutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.input_shape = kwargs["input_shape"]
        self.encoder = nn.Sequential(
            nn.Linear(self.input_shape, encoder_output_size),
            nn.ReLU(True),
            nn.Linear(encoder_output_size, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True),
            nn.Linear(8, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(True),
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Linear(16, 32),
            nn.ReLU(True),
            nn.Linear(decoder_output_size, self.input_shape), 
            nn.Tanh()
        )

    def forward(self, features):
        encoder_features = self.encoder(features)
        decoder_features = self.decoder(encoder_features)
        return decoder_features

    def setup_training(self, input_data, test_data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataloader = DataLoader(input_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        model = SCAutoEncoder(input_shape=self.input_shape).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # mean-squared error loss
        criterion = nn.MSELoss()

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
        for te_d in test:
            p_data = self.encoder.forward(torch.tensor(np.array(te_d)))
            print(p_data)
