import torch.nn as nn
from models.KANLinear import KANLinear
import torch
import torch.nn.init as init
from timm.layers import trunc_normal_
import math
class LongTermFetaureExtractor(nn.Module):
    def __init__(self, config):
        super(LongTermFetaureExtractor, self).__init__()
        self.config = config
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.future_steps = config['future_steps']
        self.seq_len = config['seq_len']
        self.device = config['device']
        self.batch_size = config['batch_size']
        self.lag_size = config['seq_len']
        self.lstm = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.lstm2 = nn.GRU(input_size=self.hidden_dim * 2, hidden_size=self.hidden_dim, num_layers=2, batch_first=True, bidirectional=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        return x

class ShortTermFeatureExtractor(nn.Module):
    # same for conv1d
    def __init__(self, config):
        super(ShortTermFeatureExtractor, self).__init__()
        self.config = config
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.future_steps = config['future_steps']
        self.seq_len = config['seq_len']
        self.device = config['device']
        self.batch_size = config['batch_size']
        self.lag_size = config['seq_len']
        self.conv1 = nn.Conv1d(in_channels=self.input_dim, out_channels=self.hidden_dim * 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=self.hidden_dim * 3, out_channels=self.hidden_dim, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.softmax(x)
        x = self.dropout(x)

        x = self.conv2(x)
        # x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        return x

class RecursiveBlock(torch.nn.Module):
    def __init__(self, config):
        super(RecursiveBlock, self).__init__()
        self.config = config

        self.longTermFeature = LongTermFetaureExtractor(config)
        self.shortTermFeature = ShortTermFeatureExtractor(config)

    def forward(self, x, data, depth):
        if depth == 0:
            return x
        longTermFeature = self.longTermFeature(x) + x
        shortTermFeature = self.shortTermFeature(x) + x
        x1 = self.forward(longTermFeature, data, depth - 1)
        x2 = self.forward(shortTermFeature, data, depth - 1)

        return torch.cat([x1, x2], dim=2)
    
class TreeKan(torch.nn.Module):
    def __init__(self, config):
        super(TreeKan, self).__init__()
        self.config = config
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.future_steps = config['future_steps']
        self.seq_len = config['seq_len']
        self.batch_size = config['batch_size']
        self.kan_grid_size = config['kan_grid_size']
        self.recursive_block = RecursiveBlock(config)
        self.lstm = nn.GRU(input_size=8 * self.hidden_dim, hidden_size=self.hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.activation = nn.ReLU()
        self.kan = KANLinear(2 * self.hidden_dim * self.seq_len, self.output_dim * self.future_steps, grid_size=self.kan_grid_size)


    def forward(self, x):
        x = self.recursive_block(x, x, 3)
        # print(x.shape)
        x, _ = self.lstm(x)
        x = self.activation(x)
        final_input = x.reshape(self.batch_size, -1)
        final_output = self.kan(final_input)
        return final_output
