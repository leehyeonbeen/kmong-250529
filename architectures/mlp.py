import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect


class MLP(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_h: int = 128, n_hl: int = 2):
        super().__init__()
        self.init_kwargs = {}
        args = inspect.getfullargspec(self.__init__)
        for k, v in zip(args.annotations.keys(), args.args[1:]):
            self.init_kwargs[k] = locals()[k]

        self.n_input = n_input
        self.n_h = n_h
        self.n_hl = n_hl
        self.n_output = n_output

        self.input_layer = nn.Linear(self.n_input, self.n_h)
        self.output_layer = nn.Linear(self.n_h, self.n_output)
        self.hidden_layers = nn.ModuleList()
        for _ in range(self.n_hl - 1):
            self.hidden_layers.append(nn.Linear(self.n_h, self.n_h))

    def forward(self, x):
        h = F.relu(self.input_layer(x))
        for l in self.hidden_layers:
            h = F.relu(l(h))
        out = F.softmax(self.output_layer(h), dim=-1)
        return out


if __name__ == "__main__":
    pass
