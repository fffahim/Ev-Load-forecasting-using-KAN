import pandas as pd
import torch

class DataPreProcess:
    def __init__(self, config):
        self.config = config

    def get_data(self, path):
        data = pd.read_csv(path)
        data = data[(data['Start'] >= self.config['start']) & (data['Start'] <= self.config['end'])]
        return data
    
    def get_peak_hours(self, data):
        data['hour'] = pd.to_datetime(data['Start']).dt.hour

        for index, i in data.iterrows():
            level = 0
            if i['Week Day'] == 1:
                if i['hour'] in self.config['peak_hour_week_l0']:
                    level = 0
                elif i['hour'] in self.config['peak_hour_week_l1']:
                    level = 1
                else:
                    level = 2
            else:
                if i['hour'] in self.config['peak_hour_weekend_l0']:
                    level = 0
                elif i['hour'] in self.config['peak_hour_weekend_l1']:
                    level = 1
                else:
                    level = 2
            data.at[index, 'peak_level'] = level
        return data
    
    def process_data(self):
        data = self.get_data(self.config['path'])
        data = self.get_peak_hours(data)
        return data
    
    def save_processed_data(self):
        data = self.process_data()
        data.to_csv(self.config['save_path'], index=False)

class DataPostProcess:
    def __init__(self, config):
        self.config = config

    def post_process(self, data, min_col, max_col):
        data[data < 0] = 0
        data = self.reconstruct_data(data, self.config['stride'], self.config['future_steps'])
        data = self.reverse_normalize(data, min_col, max_col)
        return data
    
    def save_data(self, data, path):
        pd.DataFrame(data, columns=['Energy']).to_csv(path, index=False)

    def reverse_normalize(self, data, min_col, max_col):
        data = data * (max_col[0].item() - min_col[0].item()) + min_col[0].item()
        return data
    
    def reconstruct_data(self, data, stride, future_step):
        total_output = future_step + (data.shape[0] -1) * stride
        reconstructed_data = torch.zeros(size=(total_output,), dtype=torch.float32)
        count_array = torch.zeros(size=(total_output,), dtype=torch.float32)

        for i in range(data.shape[0]):
            start = i * stride
            end = start + future_step
            reconstructed_data[start:end] += data[i]
            count_array[start:end] += 1

        reconstructed_data /= count_array
        return reconstructed_data

