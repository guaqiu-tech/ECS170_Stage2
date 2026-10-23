from local_code.base_class.method import method
import torch
from torch import nn

class Method_MLP(method, nn.Module):
    data = None
    max_epoch = 20
    learning_rate = 1e-3
    batch_size = 256

    def __init__(self, mName='MLP', mDescription='Tunable MLP',
                 use_deep=False, use_bn_dropout=False):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        layers = []
        # First hidden layer: 784 input features -> 512 hidden units
        layers.append(nn.Linear(784, 512))
        if use_bn_dropout:
            # Add BatchNorm and Dropout to prevent overfitting
            layers.append(nn.BatchNorm1d(512))
            layers.append(nn.Dropout(0.2))
        layers.append(nn.ReLU())

        # Conditional: Add a second hidden layer (Deep Network)
        if use_deep:
            layers.append(nn.Linear(512, 256))
            if use_bn_dropout:
                layers.append(nn.BatchNorm1d(256))
                layers.append(nn.Dropout(0.2))
            layers.append(nn.ReLU())
            # Output layer: 256 -> 10 classes
            layers.append(nn.Linear(256, 10))
        else:
            # Output layer if only 1 hidden layer: 512 -> 10 classes
            layers.append(nn.Linear(512, 10))

        # Combine all layers into a sequential container
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass: feed data through the network
        return self.net(x)

    def train(self, X, y):
        # Define loss function and optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        #optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

        loss_function = nn.CrossEntropyLoss()
        #loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)

        X_t = torch.FloatTensor(X)
        y_t = torch.LongTensor(y)

        # Create a DataLoader for mini-batch training
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.loss_history = []
        for epoch in range(self.max_epoch):
            # Set model to training mode
            nn.Module.train(self, True)
            epoch_loss = 0.0

            # Iterate through mini-batches
            for xb, yb in loader:
                y_pred = self.forward(xb)
                loss = loss_function(y_pred, yb)

                # Standard PyTorch gradient descent steps
                optimizer.zero_grad()  # Clear previous gradients
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

                epoch_loss += loss.item() * xb.size(0)

            # Calculate average loss for the epoch
            epoch_loss /= len(X_t)
            self.loss_history.append(epoch_loss)
            print(f'Epoch: {epoch} Loss: {epoch_loss:.4f}')

        # Save the trained model weights (Optional/Bonus)
        torch.save(self.state_dict(), 'my_mlp_model.pth')

    def test(self, X):
        # Set model to evaluation mode (disable Dropout)
        nn.Module.train(self, False)
        with torch.no_grad():  # Disable gradient calculation for testing
            X_t = torch.FloatTensor(X)
            y_pred = self.forward(X_t)
        # Return the class with the highest probability
        return y_pred.max(1)[1]

    def run(self):
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        # Return predictions, true labels, and loss history
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'], 'loss_history': self.loss_history}