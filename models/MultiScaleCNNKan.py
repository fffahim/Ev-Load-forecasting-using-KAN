import torch
import torch.nn as nn
from models.KANLinear import KANLinear

class MultiScaleCNN(nn.Module):
    def __init__(self, config):
        super(MultiScaleCNN, self).__init__()
        self.config = config
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.kan_grid_size = config['kan_grid_size']
        self.future_steps = config['future_steps']
        
        # First scale: Small kernel (short-term patterns)
        self.conv1 = nn.Conv1d(in_channels=self.input_dim, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Second scale: Medium kernel (medium-term patterns)
        self.conv2 = nn.Conv1d(in_channels=self.input_dim, out_channels=64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Third scale: Large kernel (long-term patterns)
        self.conv3 = nn.Conv1d(in_channels=self.input_dim, out_channels=64, kernel_size=7, padding=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=self.hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)

        self.residual = nn.Conv1d(in_channels=self.input_dim, out_channels=64, kernel_size=1, padding=0)
        
        # Fully connected layers
        self.kan = KANLinear(self.hidden_dim * config['seq_len'] // 2, self.output_dim * self.future_steps, grid_size=self.kan_grid_size)
        self.fc1 = nn.Linear(64 * 3, 128)  # Concatenated features from all scales
        # self.fc2 = nn.Linear(128, output_size)  # Final output layer
    
    def forward(self, x):
        # x shape: (batch_size, input_channels, seq_length)
        x = x.permute(0, 2, 1)
        residual = torch.mean(self.residual(x), dim=-1) 
        # Process through each convolution and pooling layer
        out1 = self.pool1(torch.tanh(self.conv1(x)))  # First scale
        out2 = self.pool2(torch.tanh(self.conv2(x)))  # Second scale
        out3 = self.pool3(torch.tanh(self.conv3(x)))  # Third scale
        
        # Concatenate features along the channel dimension
        # out = torch.cat((out1, out2, out3), dim=1)  # Concatenate along channels
        out = out1.mul(out2)
        out = out.mul(out3)
        out = out + residual.unsqueeze(-1)

        out = out.permute(0, 2, 1)
        out, _ = self.lstm1(out)
        out = self.dropout(out)
        # Flatten for fully connected layers
        out = out.reshape(self.config['batch_size'], -1)
        out = self.kan(out)
        
        return out
