import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim

# Dummy model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    state_dict = dict(zip(model.state_dict().keys(), parameters))
    model.load_state_dict({k: torch.tensor(v) for k, v in state_dict.items()})

class Client(fl.client.NumPyClient):
    def __init__(self, model):
        self.model = model
        self.opt = optim.SGD(self.model.parameters(), lr=0.01)
    
    def get_parameters(self, config): return get_parameters(self.model)
    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        # Dummy training
        for _ in range(1):
            self.opt.zero_grad()
            loss = self.model(torch.randn(10)).sum()
            loss.backward()
            self.opt.step()
        return get_parameters(self.model), 1, {}

    def evaluate(self, parameters, config):
        return 0.0, 1, {}

def run_fl_client():
    model = SimpleModel()
    fl.client.start_numpy_client("localhost:8080", client=Client(model))
