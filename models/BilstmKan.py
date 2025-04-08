import torch.nn as nn
from models.KANLinear import KANLinear
import torch
import torch.nn.init as init
from timm.layers import trunc_normal_
import math

class BiLSTMWithKAN(nn.Module):
    def __init__(self, config, input_dim, hidden_dim, output_dim, num_layers=1, kan_grid_size=10):
        super(BiLSTMWithKAN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.config = config
        self.future_steps = config['future_steps']

        # BiLSTM for sequence modeling
        self.bilstm1 = nn.LSTM(input_dim, input_dim // 2, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.normalize = nn.LayerNorm([256, 20, hidden_dim * 2])
        self.cnn1 = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        self.cnnnorm = nn.BatchNorm1d(input_dim)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        self.bilstm2 = nn.LSTM(input_dim, input_dim // 2, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.normalize2 = nn.LayerNorm([256, 20, hidden_dim * 2])
        self.cnn2 = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        self.cnnnorm2 = nn.BatchNorm1d(input_dim)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        self.bilstm3 = nn.LSTM(input_dim, input_dim // 2, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.normalize3 = nn.LayerNorm([256, 20, hidden_dim * 2])
        self.cnn3 = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        self.cnnnorm3 = nn.BatchNorm1d(input_dim)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        self.lstm_final = nn.LSTM(input_dim, hidden_dim // 2, num_layers=num_layers, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(0.2)
        self.activation = nn.LeakyReLU()

        # KAN layer for feature enhancement
        self.kan = KANLinear((hidden_dim) * self.config['seq_len'], output_dim * self.future_steps, grid_size=kan_grid_size)
        self.linear = nn.Linear((hidden_dim) * self.config['seq_len'], output_dim * self.future_steps)

    def forward(self, x):
        # BiLSTM
        data = x.clone()

        lstm_out, _ = self.bilstm1(x)
        # lstm_out = self.normalize(lstm_out)
        lstm_out = self.activation(lstm_out)
        lstm_out = self.dropout(lstm_out)
        cnn_out = self.cnn1(x.permute(0, 2, 1))
        cnn_out = self.cnnnorm(cnn_out)
        cnn_out = self.activation(cnn_out)
        cnn_out = self.maxpool1(cnn_out)

        lstm_out = lstm_out + data
        cnn_out = cnn_out.permute(0, 2, 1) + data
        lstm_out = lstm_out + cnn_out

        # lstm_out = self.normalize(lstm_out)

        lstm_out2, _ = self.bilstm2(lstm_out)
        lstm_out2 = self.activation(lstm_out2)
        lstm_out2 = self.dropout(lstm_out2)
        cnn_out2 = self.cnn2(lstm_out.permute(0, 2, 1))
        cnn_out2 = self.cnnnorm2(cnn_out2)
        cnn_out2 = self.activation(cnn_out2)
        cnn_out2 = self.maxpool2(cnn_out2)

        lstm_out2 = lstm_out2 + data
        cnn_out2 = cnn_out2.permute(0, 2, 1) + data
        lstm_out = lstm_out2 + cnn_out2

        lstm_out3, _ = self.bilstm3(lstm_out)
        lstm_out3 = self.activation(lstm_out3)
        lstm_out3 = self.dropout(lstm_out3)
        cnn_out3 = self.cnn3(lstm_out.permute(0, 2, 1))
        cnn_out3 = self.cnnnorm3(cnn_out3)
        cnn_out3 = self.activation(cnn_out3)
        cnn_out3 = self.maxpool3(cnn_out3)

        lstm_out3 = lstm_out3 + data
        cnn_out3 = cnn_out3.permute(0, 2, 1) + data
        lstm_out = lstm_out3 + cnn_out3

        # lstm_out = self.normalize2(lstm_out)
        lstm_out, _ = self.lstm_final(lstm_out)
        lstm_out = self.activation(lstm_out)
        lstm_out = self.dropout(lstm_out)

        lstm_out_last = lstm_out.reshape(self.config['batch_size'], -1)
        output = self.kan(lstm_out_last)

        return output

class SelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        """
        Args:
            input_dim (int): Dimension of the input features.
            attention_dim (int): Dimension of the attention mechanism.
        """
        super(SelfAttention, self).__init__()
        # Learnable projection matrices
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.softmax = nn.Softmax(dim=1)
        
        # Scaling factor for dot product
        self.scale = attention_dim ** 0.5  # Square root of attention_dim
    
    def forward(self, x):
        # Compute query, key, and value projections
        Q = self.query(x)  # Shape: (batch_size, seq_len, attention_dim)
        K = self.key(x)    # Shape: (batch_size, seq_len, attention_dim)
        V = self.value(x)  # Shape: (batch_size, seq_len, attention_dim)
        
        # Compute scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # Shape: (batch_size, seq_len, seq_len)
        attention_weights = self.softmax(attention_scores)  # Normalize across seq_len (last dimension)

        # Weighted sum of value vectors
        attention_output = torch.matmul(attention_weights, V)  # Shape: (batch_size, seq_len, attention_dim)
        
        return attention_output  # Return both output and weights if needed


class DPNnBlock(torch.nn.Module):
    def __init__(self, config, input_dim):
        super(DPNnBlock, self).__init__()
        self.config = config
        self.input_dim = input_dim
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.kan_grid_size = config['kan_grid_size']
        self.attention_heads = 2
        self.kernel_size = 3
        self.stride = 1
        self.padding = 1

        #First phase
        self.bilstm1 = nn.GRU(self.input_dim, self.hidden_dim, batch_first=True, bidirectional=True, num_layers=1)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)
        self.kanbilstm = KANLinear(self.hidden_dim * self.config['seq_len'], self.hidden_dim * self.config['seq_len'], grid_size=self.kan_grid_size)
        # self.attention_lstm1 = SelfAttention(2 * self.hidden_dim, 2 * self.hidden_dim)

        self.cnn1 = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=3, padding=1, dilation=1)
        self.normcnn = nn.BatchNorm1d(self.hidden_dim, momentum=0.9)
        self.maxpool1 = nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride, padding=1)
        self.softmax1 = nn.Softmax(dim=1)
        self.kancnn = KANLinear(self.hidden_dim * self.config['seq_len'], self.hidden_dim * self.config['seq_len'], grid_size=self.kan_grid_size)
        # self.attention_cnn1 = SelfAttention(2 * self.hidden_dim, 2 * self.hidden_dim)

        # self.attention1_phase1 = nn.MultiheadAttention(embed_dim=2 * self.hidden_dim, num_heads=self.attention_heads, batch_first=True)
        # self.attention2_phase1 = nn.MultiheadAttention(embed_dim=2 * self.hidden_dim, num_heads=self.attention_heads, batch_first=True)

        #Second phase
        self.bilstm2 = nn.GRU(1 * self.hidden_dim, self.input_dim, batch_first=True, bidirectional=True, num_layers=1)
        self.activation2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(0.1)
        # self.kanbilstm2 = KANLinear(self.input_dim * self.config['seq_len'], self.input_dim * self.config['seq_len'], grid_size=self.kan_grid_size)
        self.attention_lstm2 = SelfAttention(self.input_dim, self.input_dim)

        self.cnn2 = nn.Conv1d(1 * self.hidden_dim, self.input_dim, kernel_size=3, padding=1, dilation=1, stride=1)
        self.normcnn2 = nn.BatchNorm1d(self.input_dim, momentum=0.9)
        self.maxpool2 = nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride, padding=1)
        self.softmax2 = nn.Softmax(dim=1)
        # self.kancnn2 = KANLinear(self.input_dim * self.config['seq_len'], self.input_dim * self.config['seq_len'], grid_size=self.kan_grid_size)
        self.attention_cnn2 = SelfAttention(self.input_dim, self.input_dim)
        
        # self.attention1_phase2 = nn.MultiheadAttention(embed_dim=self.input_dim, num_heads=self.attention_heads, batch_first=True)
        # self.attention2_phase2 = nn.MultiheadAttention(embed_dim=self.input_dim, num_heads=self.attention_heads, batch_first=True)

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
    

    def forward(self, x, x1, x2):
        #First phase
        data = x.clone()

        lstm_out1, _ = self.bilstm1(x1)
        lstm_out1 = (lstm_out1[:, :, :self.hidden_dim] + lstm_out1[:, :, self.hidden_dim:]) / 2
        # lstm_out1 = self.activation(lstm_out1)
        lstm_out1 = self.dropout(lstm_out1)
        # lstm_out1 = lstm_out1.permute(0, 2, 1)
        lstm_kan_input1 = lstm_out1.reshape(self.config['batch_size'], -1) 
        lstm_kan_out1 = self.kanbilstm(lstm_kan_input1)
        lstm_kan_out1 = lstm_kan_out1.reshape(self.config['batch_size'], self.config['seq_len'], -1)
        # lstm_kan_out1 = lstm_kan_out1 + data
        # lstm_attention1 = self.attention_lstm1(lstm_kan_out1)

        cnn_out1 = self.cnn1(x2.permute(0, 2, 1))
        cnn_out1 = self.normcnn(cnn_out1)
        # cnn_out1 = self.activation(cnn_out1)
        # cnn_out1 = self.maxpool1(cnn_out1)
        # cnn_out1 = self.softmax1(cnn_out1)
        cnn_out1 = self.dropout(cnn_out1)
        cnn_out1 = cnn_out1.permute(0, 2, 1)
        cnn_kan_input1 = cnn_out1.reshape(self.config['batch_size'], -1)
        cnn_kan_out1 = self.kancnn(cnn_kan_input1)
        cnn_kan_out1 = cnn_kan_out1.reshape(self.config['batch_size'], self.config['seq_len'], -1)
        # cnn_kan_out1 = cnn_kan_out1 + data
        # cnn_attention1 = self.attention_cnn1(cnn_kan_out1)

        # cnn_out1 = torch.cat((cnn_out1, lstm_kan_out1), dim=2)
        # lstm_out1 = torch.cat((lstm_out1, cnn_kan_out1), dim=2)
        cnn_out1 = cnn_out1 * lstm_kan_out1
        lstm_out1 = lstm_out1 * cnn_kan_out1

        #Second phase
        lstm_out2, _ = self.bilstm2(lstm_out1)
        # lstm_out2 = self.activation2(lstm_out2)
        lstm_out2 = (lstm_out2[:, :, :self.input_dim] + lstm_out2[:, :, self.input_dim:]) / 2
        lstm_out2 = self.dropout2(lstm_out2)
        # lstm_kan_out2 = lstm_out2
        # lstm_kan_input2 = lstm_out2.reshape(self.config['batch_size'], -1)
        # lstm_kan_out2 = self.kanbilstm2(lstm_kan_input2)
        # lstm_kan_out2 = lstm_kan_out2.reshape(self.config['batch_size'], self.config['seq_len'], -1)
        # # lstm_kan_out2, _ = self.attention1_phase2(lstm_kan_out2, lstm_kan_out2, lstm_kan_out2)
        # lstm_kan_out2 = lstm_kan_out2 - lstm_out2
        # lstm_attention2 = self.attention_lstm2(lstm_out2)

        cnn_out2 = self.cnn2(cnn_out1.permute(0, 2, 1))
        cnn_out2 = self.normcnn2(cnn_out2)
        # cnn_out2 = self.activation2(cnn_out2)
        # cnn_out2 = self.maxpool2(cnn_out2)
        # cnn_out2 = self.softmax2(cnn_out2)
        cnn_out2 = self.dropout2(cnn_out2)
        cnn_out2 = cnn_out2.permute(0, 2, 1)
        # cnn_kan_input2 = cnn_out2.reshape(self.config['batch_size'], -1)
        # cnn_kan_out2 = self.kancnn2(cnn_kan_input2)
        # cnn_kan_out2 = cnn_kan_out2.reshape(self.config['batch_size'], self.config['seq_len'], -1)
        # # # cnn_kan_out2, _ = self.attention2_phase2(cnn_kan_out2, cnn_kan_out2, cnn_kan_out2)
        # cnn_kan_out2 = cnn_out2 - cnn_kan_out2
        # cnn_kan_out2 = cnn_out2
        # cnn_attention2 = self.attention_cnn2(cnn_out2)

        # shared_out2 = torch.cat((lstm_kan_out2, cnn_kan_out2), dim=2)
        # lstm_out2 = lstm_out2 + data - cnn_out2
        # cnn_out2 = cnn_out2  + data - lstm_out2
        # cnn_out2 = torch.cat((cnn_out2, lstm_out2, data), dim=2)
        # lstm_out2 = torch.cat((lstm_out2, cnn_out2, data), dim=2)

        lstm_out3 = lstm_out2  + data
        cnn_out3 = cnn_out2  + data

        return lstm_out3, cnn_out3


class DPNnKAN(nn.Module):
    def __init__(self, config):
        super(DPNnKAN, self).__init__()
        self.config = config
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.kan_grid_size = config['kan_grid_size']

        self.dpnnblock1 = DPNnBlock(config, config['input_dim'])
        self.dpnnblock2 = DPNnBlock(config, config['input_dim'])
        self.dpnnblock3 = DPNnBlock(config, config['input_dim'])

        self.lstm = nn.GRU(2 * self.input_dim, self.input_dim, batch_first=True, bidirectional=True, num_layers=1)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)
        # self.kan = KANLinear(self.input_dim * 2 * self.config['seq_len'], self.output_dim * config["future_steps"], grid_size=self.kan_grid_size)
        self.linear = nn.Linear(self.input_dim * 2 * self.config['seq_len'], self.output_dim * config["future_steps"])

    def forward(self, x):
        lstm_out, cnn_out = self.dpnnblock1(x, x, x)
        lstm_out, cnn_out = self.dpnnblock2(x, lstm_out, cnn_out)
        # lstm_out, cnn_out = self.dpnnblock3(x, lstm_out, cnn_out)
        final_input = torch.cat((lstm_out, cnn_out), dim=2)
        # final_input = lstm_out + cnn_out
        # final_input = final_input.permute(0, 2, 1)
        final_output, _ = self.lstm(final_input)
        # final_output, _ = self.lstm2(final_output)
        final_output = self.dropout(final_output)
        # final_output = self.activation(final_output)
        # final_output = final_output.permute(0, 2, 1)
        # final_output, _ = self.lstm(final_input)
        final_output = final_output.reshape(self.config['batch_size'], -1)
        final_output = self.linear(final_output)
        
        return final_output