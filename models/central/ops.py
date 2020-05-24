import torch
import torch.nn as nn
import torch.nn.functional as F


class activ(nn.Module):
    def __init__(self, args):
        super(activ, self).__init__()
        self.activation = args.activation
        if args.activation == "LeakyReLU":
            self.act = torch.nn.LeakyReLU()
        elif args.activation == "ELU":
            self.act = torch.nn.ELU()
        elif args.activation == "ReLU":
            self.act = torch.nn.ReLU()
        elif args.activation == "Tanh":
            self.act = torch.nn.Tanh()
        elif args.activation == "Sigmoid":
            self.act = torch.nn.Sigmoid()
        elif args.activation == "Swish":
            self.beta = nn.Parameter(torch.tensor(0.5))
            self.act = torch.nn.Sigmoid()
        else:
            print("WARNING: REQUIRED ACTIVATION IS NOT DEFINED")

    def forward(self, x):
        if self.activation == "Swish":
            return self.act(self.beta * x) * x
        else:
            return self.act(x)
