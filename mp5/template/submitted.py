import torch
import torch.nn as nn


def create_sequential_layers():
    """
    Task: Create neural net layers using nn.Sequential.

    Requirements: Return an nn.Sequential object, which contains:
        1. a linear layer (fully connected) with 2 input features and 3 output features,
        2. a sigmoid activation layer,
        3. a linear layer with 3 input features and 5 output features.
    """
    block = nn.Sequential(nn.Linear(2,3), nn.Sigmoid(), nn.Linear(3,5))
    return block


def create_loss_function():
    """
    Task: Create a loss function using nn module.

    Requirements: Return a loss function from the nn module that is suitable for
    multi-class classification.
    """
    return nn.CrossEntropyLoss()


class NeuralNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize your neural network here.
        """
        super().__init__()
        ################# Your Code Starts Here #################
        self.input = nn.Linear(2883, 300)
        self.activation = nn.LeakyReLU()
        self.hidden1 = nn.Linear(300, 100)
        self.output = nn.Linear(100, 5)

        ################## Your Code Ends here ##################

    def forward(self, x):
        """
        Perform a forward pass through your neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """
        ################# Your Code Starts Here #################
        x_temp1 = self.input(x)
        x_temp2 = self.activation(x_temp1)
        x_temp3 = self.hidden1(x_temp2)
        y = self.output(x_temp3)

        return y
        ################## Your Code Ends here ##################


def train(train_dataloader, epochs):
    """
    The autograder will call this function and compute the accuracy of the returned model.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        test_dataloader:    a dataloader for the testing set and labels
        epochs:             the number of times to iterate over the training set

    Outputs:
        model:              trained model
    """

    ################# Your Code Starts Here #################
    """
    Implement backward propagation and gradient descent here.
    """
    # Create an instance of NeuralNet, a loss function, and an optimizer

    model = NeuralNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 9e-5, weight_decay = 1e-7)
    for epoch in range(epochs):
        for features, labels in train_dataloader:
            model.train()
            y_pred = model(features)
            y_true = labels
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()
            
    ################## Your Code Ends here ##################

    return model
