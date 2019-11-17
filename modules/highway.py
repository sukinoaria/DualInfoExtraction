import torch
import torch.nn as nn

class Highway(nn.Module):
    def __init__(self, input_size, output_size, num_layers, f):
        super().__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(output_size if i else input_size, output_size) for i in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(output_size if i else input_size, output_size) for i in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(output_size if i else input_size, output_size) for i in range(num_layers)])
        self.f = f

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_layers):
            nn.init.constant_(self.linear[i].bias, 0)
            nn.init.xavier_normal_(self.linear[i].weight)
            nn.init.constant_(self.nonlinear[i].bias, 0)
            nn.init.xavier_normal_(self.linear[i].weight)
            nn.init.constant_(self.gate[i].bias, 0)
            nn.init.xavier_normal_(self.gate[i].weight)

    def forward(self, x):
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x