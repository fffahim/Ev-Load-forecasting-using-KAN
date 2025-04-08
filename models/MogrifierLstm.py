import torch
import torch.nn as nn

class Mogrifier_LSTM(nn.Module):
    def __init__(self, config, layer_size = 2):
        super(Mogrifier_LSTM, self).__init__()

        self.config = config
        self.input_size = config['input_dim']
        self.hidden_size = config['hidden_dim']
        self.future_step = config['future_steps']
        self.batch_size = config['batch_size']
        self.lag_size = config['seq_len']
        self.device = config['device']
        self.output_dim = config['output_dim']

        self.mogrifier_lstm_layer1 = MogrifierLSTMCell(config, layer_size)
        self.mogrifier_lstm_layer2 = MogrifierLSTMCell(config, layer_size)

        # Backward LSTM layer
        self.mogrifier_lstm_layer1_backward = MogrifierLSTMCell(config, layer_size)
        self.mogrifier_lstm_layer2_backward = MogrifierLSTMCell(config, layer_size)

        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size * self.lag_size, self.future_step * self.output_dim)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        h1, c1 = [torch.zeros(self.batch_size, self.hidden_size).to(self.device), torch.zeros(self.batch_size, self.hidden_size).to(self.device)]
        h2, c2 = [torch.zeros(self.batch_size, self.hidden_size).to(self.device), torch.zeros(self.batch_size, self.hidden_size).to(self.device)]
        
        h1_backward, c1_backward = torch.zeros(self.batch_size, self.hidden_size).to(self.device), torch.zeros(self.batch_size, self.hidden_size).to(self.device)
        h2_backward, c2_backward = torch.zeros(self.batch_size, self.hidden_size).to(self.device), torch.zeros(self.batch_size, self.hidden_size).to(self.device)

        hidden_states_forward = []
        hidden_states_backward = []

        outputs = []
        for i in range(self.lag_size):
            tempx = x[:, i]
            h1,c1 = self.mogrifier_lstm_layer1(tempx, (h1, c1))     
            hidden_states_forward.append(h1.unsqueeze(1))

        hidden_states_forward = torch.cat(hidden_states_forward, dim=1)

        outputs = self.linear(hidden_states_forward.view(self.batch_size, -1))  # Only take the last time step output
        outputs = outputs.view(self.batch_size, self.future_step)  
        
        return self.relu(outputs)
    
class MogrifierLSTMCell(nn.Module):

    def __init__(self, config, mogrify_steps):
        super(MogrifierLSTMCell, self).__init__()
        self.mogrify_steps = mogrify_steps
        self.config = config
        self.input_size = config['input_dim']
        self.hidden_size = config['hidden_dim']

        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.mogrifier_list = nn.ModuleList([nn.Linear(self.hidden_size, self.input_size)])
        for i in range(1, mogrify_steps):
            if i % 2 == 0:
                self.mogrifier_list.extend([nn.Linear(self.hidden_size, self.input_size)])
            else:
                self.mogrifier_list.extend([nn.Linear(self.input_size, self.hidden_size)])

    def mogrify(self, x, h):
        for i in range(self.mogrify_steps):
            if (i+1) % 2 == 0: 
                h = (2*torch.sigmoid(self.mogrifier_list[i](x))) * h
            else:
                x = (2*torch.sigmoid(self.mogrifier_list[i](h))) * x
        return x, h

    def forward(self, x, states):
        ht, ct = states
        x, ht = self.mogrify(x, ht)
        ht, ct = self.lstm(x, (ht, ct))
        return ht, ct