import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy import stats as stat

class Dataset:
    def __init__(self, x, seq_len, future_step):
        self.seq_len = seq_len
        self.x = x
        self.future_step = future_step
        self.X, self.y = self.create_sequences(x)

    def create_sequences(self, data):
        X, y = [], []
        for i in range(0, len(data) - self.seq_len - self.future_step + 1):
            X.append(data[i:i + self.seq_len])
            y.append(data[i + self.seq_len:i + self.seq_len + self.future_step, 0])
        X = torch.stack(X)
        y = torch.stack(y)

        return X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class DataManager:
    def __init__(self, config):
        self.config = config

    def spilit_data(self, data):
        split_1 = int(0.70 * len(data))
        split_2 = int(0.70 * len(data))
        train = data[:split_1]
        validation = data[split_1:]
        test = data[split_2:]
        return train, validation, test
        
    def get_data(self):
        future_steps = self.config['future_steps']
        seq_len = self.config['seq_len']

        data = pd.read_csv(self.config['path'])
        # take the rolling mean of the previous 24 hours
        data['mean'] = data['Energy'].rolling(window=20, min_periods=1).mean()
        data['std'] = data['Energy'].rolling(window=20, min_periods=1).std()
        data['std'] = data['std'].fillna(0)
        data.drop(["Start"], axis=1, inplace=True)
        data['period'] = data['hour'].apply(
            lambda hour: 0 if 0 <= hour < 6 else
                        1 if 6 <= hour < 12 else
                        2 if 12 <= hour < 18 else
                        3)
        data = data[self.config['columns']]
        data['peak_level'] = data['peak_level'].apply(lambda x: 1 if x > 0 else 0)
        # for col in data.columns[1:4]:
        #     data[col] = stat.zscore(data[col])
        for i in range(1, self.config['future_steps']):
            data[f'energy_lag_{i}'] = data['Energy'].shift(i)

        # Fill NaN values with 0
        data.fillna(0, inplace=True)
        data = torch.tensor(data.values, dtype=torch.float32, device=self.config['device'])

        min_col = torch.min(data, dim=0).values
        max_col = torch.max(data, dim=0).values
        # mean_col = torch.mean(data, dim=0)
        # std_col = torch.std(data, dim=0)
        data_normalized = (data - min_col) / (max_col - min_col)
        # data_standardized = (data - mean_col) / std_col

        train_normalized, val_normalized, test_normalized = self.spilit_data(data_normalized)

        # train_normalized, train_min, train_max = self.normalize_data(train)
        # val_normalized, val_min, val_max = self.normalize_data(val, train_min)
        # test_normalized, test_min, test_max = self.normalize_data(test, train_min)

        train_dataloader = DataLoader(Dataset(train_normalized, seq_len, future_steps), batch_size=self.config['batch_size'], shuffle=True, drop_last=True)
        test_dataloader = DataLoader(Dataset(test_normalized, seq_len, future_steps), batch_size=self.config['batch_size'], shuffle=False, drop_last=True)
        val_dataloader = DataLoader(Dataset(val_normalized, seq_len, future_steps), batch_size=self.config['batch_size'], shuffle=False, drop_last=True)
        return train_dataloader, val_dataloader, test_dataloader, min_col, max_col
    
    def normalize_data(self, data, mean_columns=None):
        min_columns = torch.min(data, dim=0).values
        max_columns = torch.max(data, dim=0).values

        if mean_columns is None:
            mean_columns = torch.mean(data, dim=0)
        std_columns = torch.std(data, dim=0)
        data = (data - mean_columns)

        # data = (data - min_columns) / (max_columns - min_columns)
        output_column_min = mean_columns[0].item()
        output_column_max = std_columns[0].item()
        return data, mean_columns, std_columns
