import torch
from models.BilstmKan import BiLSTMWithKAN, DPNnKAN
from models.MultiScaleCNNKan import MultiScaleCNN
from models.MogrifierLstm import Mogrifier_LSTM
from models.SCINet import SCINet
from models.TreeKan import TreeKan
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import pdb

import argparse
from DataProcess import DataPreProcess, DataPostProcess
from DataManager import DataManager
from Wrapper import Wrapper
from DataVisualization import DataVisualization

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--process_data', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='caltech')
    parser.add_argument('--model', type=str, default='bilstmkan')
    return parser.parse_args()

def main():
    args = vars(parse_args())
    process_data = args.get('process_data')
    dataset = args.get('dataset')
    model_arg = args.get('model')

    config = json.load(open('config/data.json'))
    config = config[dataset]

    model_config = json.load(open('config/model.json'))[model_arg]
    model_config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if process_data:
        data_process = DataPreProcess(config)
        data_process.save_processed_data()

    data_manager = DataManager(model_config)
    wrapper = Wrapper(model_config)
    data_visualization = DataVisualization()
    data_post_process = DataPostProcess(model_config)

    train_dataloader, val_dataloader, test_dataloader, \
                min_col, max_col = data_manager.get_data()
    
    # i = 0
    # for x, y in test_dataloader:
    #     print(x.shape, y.shape)
    #     i += 1
    
    # print(i)

    if model_arg == 'bilstmkan':
        # model = BiLSTMWithKAN(model_config, model_config['input_dim'], model_config['hidden_dim'], model_config['output_dim'], model_config['num_layers'], model_config['kan_grid_size'])
        model = DPNnKAN(model_config)
        # model = MultiScaleCNN(model_config)
        # model = TreeKan(model_config)
    elif model_arg == 'mogrifierlstm':
        print('mogrifier lstm')
        model = Mogrifier_LSTM(model_config)
    elif model_arg == 'scinet':
        model = SCINet(output_len = model_config['future_steps'], input_len= model_config['seq_len'], input_dim = model_config['input_dim'], hid_size = model_config['hidden_dim'], num_stacks = model_config['stacks'],
                num_levels = model_config['levels'], concat_len = 0, groups = 1, kernel = 4, dropout = 0.2,
                 single_step_output_One = 0, positionalE =  False, modified = True)
        

    model.to(model_config['device'])

    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'], weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    tx, ty  = wrapper.train(model, train_dataloader, val_dataloader, optimizer, criterion, scheduler)
    ground_truth, predicted = wrapper.test(model, test_dataloader)
  
    #reverse standardization
    # ground_truth = ground_truth + mean[0].item()
    # predicted = predicted + mean[0].item()

    #reverse standardization
    # ground_truth = ground_truth * std_col[0].item() + mean_col[0].item()
    # predicted = predicted * std_col[0].item() + mean_col[0].item()

    #negetive values in the data to zero
    ground_truth = data_post_process.post_process(ground_truth.cpu().detach(), min_col, max_col)
    predicted = data_post_process.post_process(predicted.cpu().detach(), min_col, max_col)
    wrapper.evaluate(ground_truth, predicted)

    # ground_truth = ground_truth.unsqueeze(-1)[:, 0, :]
    # predicted = predicted.unsqueeze(-1)[:, 0, :]
    future_steps = model_config['future_steps']
    data_post_process.save_data(ground_truth.cpu().detach().numpy(), f'data/predicted/{dataset}_{model_arg}_{future_steps}_ground_truth3.csv')
    data_post_process.save_data(predicted.cpu().detach().numpy(), f'data/predicted/{dataset}_{model_arg}_{future_steps}_predicted3.csv')

    data_visualization.plot_forecasting_results(ground_truth, predicted)


if __name__ == "__main__":
    main()
