import torch
import numpy as np
from sklearn.metrics import r2_score
import time

class Wrapper:
    def __init__(self, config):
        self.config = config

    def train(self, model, train_dataloader, val_dataloader, optimizer, criterion, scheduler=None, val=None):
        ground_truth = torch.empty((0, self.config['output_dim'] * self.config['future_steps']), dtype=torch.float32).to(self.config['device'])
        predicted = torch.empty((0, self.config['output_dim'] * self.config['future_steps']), dtype=torch.float32).to(self.config['device'])
        max_grad_norm = 1.0
        epochs = self.config['epoch']
        start_time = time.time()
        for epoch in range(epochs):
            running_loss = 0.0
            count = 0
            model.train()
            for x, y in train_dataloader:
                x, y = x.to(self.config['device']), y.to(self.config['device'])
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                L2 = torch.sum((output - y)**2)
                # loss = loss + L2
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                running_loss += loss.item()
                count += 1
                if epoch == epochs - 1:
                    ground_truth = torch.cat((ground_truth, y), dim=0)
                    predicted = torch.cat((predicted, output), dim=0)

            if val:
                val_loss = 0.0
                count = 0
                model.eval()
                with torch.no_grad():
                    for x, y in val_dataloader:
                        if len(y.shape) == 1:
                            y = y.unsqueeze(-1)
                        x, y = x.to(self.config['device']), y.to(self.config['device'])
                        output = model(x)
                        loss = criterion(output, y)
                        L2 = torch.sum((output - y)**2)
                        # loss = loss + L2
                        val_loss += loss.item()
                        count += 1
                if scheduler is not None:
                    scheduler.step(val_loss)
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss / count}, Val Loss: {val_loss / count}')
            else:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss / count}')

        end_time = time.time()
        diff_per_epoch = (end_time - start_time) / epochs
        print(f'Training time: {end_time - start_time}, Time per epoch: {diff_per_epoch}')

        return ground_truth, predicted

    def test(self, model, test_dataloader):
        ground_truth = torch.empty((0, self.config['output_dim'] * self.config['future_steps']), dtype=torch.float32).to(self.config['device'])
        predicted = torch.empty((0, self.config['output_dim'] * self.config['future_steps']), dtype=torch.float32).to(self.config['device'])
        model.eval()

        with torch.no_grad():
            for x, y in test_dataloader:
                x, y = x.to(self.config['device']), y.to(self.config['device'])
                output = model(x)
                ground_truth = torch.cat((ground_truth, y), dim=0)
                predicted = torch.cat((predicted, output), dim=0)

        return ground_truth, predicted
    
    def evaluate(self, ground_truth, predicted):
        mse = torch.mean((ground_truth - predicted) ** 2)
        mae = torch.mean(torch.abs(ground_truth - predicted))
        r2 = r2_score(ground_truth.cpu().detach().numpy(), predicted.cpu().detach().numpy())
        rmse = torch.sqrt(mse)
        print(f'RMSE: {rmse}, MAE: {mae}, R2: {r2}')
        return mse, mae, r2