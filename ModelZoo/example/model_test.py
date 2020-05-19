from tools import *
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = []
        self.dropouts = []

        # We optimize the number of layers, hidden units in each layer and dropouts.
        n_layers = 2
        layer_units = [55.9194619944798, 9.170082514198038]
        dropout = 0.4875033619203331
        input_dim = 28 * 28
        for i in range(n_layers):
            output_dim = int(layer_units[i])
            self.layers.append(nn.Linear(input_dim, output_dim))
            self.dropouts.append(nn.Dropout(dropout))
            input_dim = output_dim

        self.layers.append(nn.Linear(input_dim, 10))

        # Assigning the layers as class variables (PyTorch requirement).
        for idx, layer in enumerate(self.layers):
            setattr(self, "fc{}".format(idx), layer)

        # Assigning the dropouts as class variables (PyTorch requirement).
        for idx, dropout in enumerate(self.dropouts):
            setattr(self, "drop{}".format(idx), dropout)
    def forward(self, data):
        data = data.view(-1, 28 * 28)
        for layer, dropout in zip(self.layers, self.dropouts):
            data = F.relu(layer(data))
            data = dropout(data)
        return F.log_softmax(self.layers[-1](data), dim=1)