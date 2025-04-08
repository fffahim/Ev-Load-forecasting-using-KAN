import matplotlib.pyplot as plt

class DataVisualization:
    def __init__(self):
        pass

    def plot_forecasting_results(self, ground_truth, predicted):
        plt.plot(ground_truth, label='Ground Truth')
        plt.plot(predicted, label='Predicted')
        plt.legend()
        plt.show()