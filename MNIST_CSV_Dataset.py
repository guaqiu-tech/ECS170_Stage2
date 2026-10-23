from local_code.base_class.dataset import dataset
import numpy as np

class MNIST_CSV_Dataset(dataset):
    def __init__(self, csv_path, dName='MNIST_CSV', dDescription='MNIST digits CSV'):
        super().__init__(dName, dDescription)
        self.csv_path = csv_path

    def load(self):
        print(f"Loading data from {self.csv_path} ...")
        # Read the CSV file into a numpy array
        data = np.loadtxt(self.csv_path, delimiter=',')

        # Extract features (pixels from column 1 to end) and normalize to [0, 1]
        X = data[:, 1:] / 255.0

        # Extract labels (the first column, values 0-9)
        y = data[:, 0]

        print(f"Data loaded! X shape: {X.shape}, y shape: {y.shape}")
        # Return as a dictionary for the Setting module
        return {'X': X, 'y': y}